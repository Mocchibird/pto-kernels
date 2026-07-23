// fused_hadamard_mxfp4_a5 — A5 (Ascend 950 / dav-c310-vec) register-resident
// fp16/bf16 Walsh-Hadamard (N=128) FUSED with MXFP4 (e2m1, block=32) quant.
//
// Per row (= one 128-lane b16 register):
//   vlds -> [7 butterfly stages in registers] -> *1/sqrt(128)
//        -> per-32-block |max| (vcmax) -> broadcast (vdup POS_LOWEST)
//        -> e8m0 = f16_exp(amax) + 110 ; mult = 2^(2-(exp-15)) (f16 power-of-two)
//        -> vmul ; f16 -> bf16 -> fp4(e2m1, PK4_B32 pack) ; store fp4 + e8m0
//
// The fp4 cast only exists as bf16->f4e2m1x2, so the final cast pivots through
// bf16; all reductions/scale math stay in f16 (bf16 has no vabs/vcmax).
//
// PERF: the row loop is manually UNROLLed UNROLL-way and every op is issued for
// all UNROLL independent rows back-to-back, so the vector pipe stays busy
// through each op's latency instead of stalling on the per-row dependency chain
// (butterfly vdintlv->vadd->vsel and the vcmax block reductions).
//
// Build: bisheng --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE
//        -DHAD_IN_BF16={0,1} [-DROWS_PER_TILE=<rows>]

#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef HAD_N
#define HAD_N 128
#endif
#ifndef HAD_LOG2N
#define HAD_LOG2N 7
#endif
static_assert(HAD_N == 128, "v1 supports N==128");
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 64
#endif
#ifndef HAD_IN_BF16
#define HAD_IN_BF16 0            // 0 => fp16 input, 1 => bf16 input
#endif
#ifndef HAD_UNROLL
#define HAD_UNROLL 8
#endif

#define MX_BLK 32
constexpr float HAD_INV = 0.08838834764831845f;   // 1/sqrt(128)

#ifdef __CCE_AICORE__
#if HAD_IN_BF16
using in_t = bfloat16_t;
#else
using in_t = half;
#endif

constexpr unsigned LANES = 128;
constexpr unsigned NBLK  = HAD_N / MX_BLK;            // 4
// vector NORM stores need 32-byte-aligned addresses; each row's e8m0 scales use
// a padded 32-byte slot (4 u16 written via alignment-exempt ONEPT stores).
constexpr unsigned SCALE_STRIDE = 32;                 // bytes/row for scales
constexpr unsigned FLAT_IN = ROWS_PER_TILE * HAD_N;
constexpr unsigned X_BYTES = FLAT_IN * sizeof(in_t);
constexpr unsigned Q_BYTES = ROWS_PER_TILE * (HAD_N / 2);
constexpr unsigned S_BYTES = ROWS_PER_TILE * SCALE_STRIDE;

constexpr unsigned W_BYTES = FLAT_IN * sizeof(half);  // WHT intermediate (f16)
constexpr unsigned aln(unsigned b) { return (b + 511u) & ~511u; }
constexpr unsigned X0 = 0;
constexpr unsigned Q0 = X0 + aln(X_BYTES);
constexpr unsigned S0 = Q0 + aln(Q_BYTES);
constexpr unsigned X1 = S0 + aln(S_BYTES);
constexpr unsigned Q1 = X1 + aln(X_BYTES);
constexpr unsigned S1 = Q1 + aln(Q_BYTES);
constexpr unsigned W0 = S1 + aln(S_BYTES);   // shared butterfly->quant scratch
constexpr unsigned UB_END = W0 + aln(W_BYTES);
static_assert(UB_END <= 192u * 1024u, "UB overflow");
static_assert(ROWS_PER_TILE % HAD_UNROLL == 0, "ROWS_PER_TILE must be multiple of HAD_UNROLL");

#ifdef __DAV_VEC__
#define DOU8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)   // butterfly unroll
#define DOU4(M) M(0) M(1) M(2) M(3)                        // quant unroll (reg-limited)

// Phase A: register-resident WHT for `rows` rows, xb -> wb (f16), 8-way unrolled.
__tf__ static AICORE void hadamard_phase(__ubuf__ in_t *xb, __ubuf__ half *wb, uint32_t rows)
{
    __VEC_SCOPE__
    {
        uint32_t la = LANES;   MaskReg pAll = CreatePredicate<half>(la);
        uint32_t l2 = LANES/2; MaskReg mlo  = CreatePredicate<half>(l2);
        vector_f16 v0,v1,v2,v3,v4,v5,v6,v7, e0,e1,e2,e3,e4,e5,e6,e7;
        vector_f16 o0,o1,o2,o3,o4,o5,o6,o7, s0,s1,s2,s3,s4,s5,s6,s7;
        for (uint16_t row = 0; row < (uint16_t)rows; row += 8) {
#if HAD_IN_BF16
            vector_bf16 w0,w1,w2,w3,w4,w5,w6,w7;
#define LD(i) vlds(w##i, xb, (uint32_t)(row+i)*LANES, NORM); vcvt(v##i, w##i, pAll, RS_DISABLE, ROUND_R);
#else
#define LD(i) vlds(v##i, xb, (uint32_t)(row+i)*LANES, NORM);
#endif
            DOU8(LD)
#undef LD
            for (uint16_t st = 0; st < (uint16_t)HAD_LOG2N; ++st) {
#define DI(i) vdintlv(e##i, o##i, v##i, v##i);
                DOU8(DI)
#undef DI
#define AD(i) vadd(s##i, e##i, o##i, pAll);
                DOU8(AD)
#undef AD
#define SU(i) vsub(v##i, e##i, o##i, pAll);   /* diffs into v (e,o dead) */
                DOU8(SU)
#undef SU
#define SE(i) vsel(v##i, s##i, v##i, mlo);    /* v = mlo? sums : diffs */
                DOU8(SE)
#undef SE
            }
#define MW(i) vmuls(v##i, v##i, (half)HAD_INV, pAll, MODE_ZEROING); \
            vsts(v##i, wb, (uint32_t)(row+i)*LANES, NORM_B16, pAll);
            DOU8(MW)
#undef MW
        }
        mem_bar(VST_VLD);   // WHT stores to W visible to the quant phase's loads
    }
}

// Phase B: MXFP4 quant of the WHT rows, wb -> {sb_q fp4, sb_s e8m0}, 8-way.
__tf__ static AICORE void quant_phase(__ubuf__ half *wb, __ubuf__ uint8_t *sb_q,
                                      __ubuf__ uint8_t *sb_s, uint32_t rows)
{
    __VEC_SCOPE__
    {
        uint32_t la = LANES; MaskReg pAll = CreatePredicate<half>(la);
        uint32_t s64 = 64; MaskReg pq = CreatePredicate<float>(s64);
        vector_u16 c1F, c32;
        vbr(c1F, (uint16_t)0x1F);
        vbr(c32, (uint16_t)32);

        // block maxes via ONE group-16 reduction (vcgmax) then pairwise combine,
        // giving the 4 block(32) maxes in lanes 0..3; broadcast to per-block 32
        // lanes with a 5-step vintlv upsample. No per-block vcmax/vdup/vsel.
        uint32_t f4 = 4; MaskReg p4 = CreatePredicate<half>(f4);
        vector_f16 v0,v1,v2,v3,v4,v5,v6,v7, a0,a1,a2,a3,a4,a5,a6,a7;
        vector_f16 cg0,cg1,cg2,cg3,cg4,cg5,cg6,cg7, ev0,ev1,ev2,ev3,ev4,ev5,ev6,ev7;
        vector_f16 od0,od1,od2,od3,od4,od5,od6,od7, bm0,bm1,bm2,bm3,bm4,bm5,bm6,bm7;
        vector_u16 bd0,bd1,bd2,bd3,bd4,bd5,bd6,bd7, mf0,mf1,mf2,mf3,mf4,mf5,mf6,mf7;
        vector_u16 lo0,lo1,lo2,lo3,lo4,lo5,lo6,lo7, hi;
        vector_bf16 cb0,cb1,cb2,cb3,cb4,cb5,cb6,cb7;
        vector_f4e2m1x2 q0,q1,q2,q3,q4,q5,q6,q7;

        for (uint16_t row = 0; row < (uint16_t)rows; row += 8) {
#define LDW(i) vlds(v##i, wb, (uint32_t)(row+i)*LANES, NORM); vabs(a##i, v##i, pAll);
            DOU8(LDW)
#undef LDW
            // 4 block(32) maxes -> lanes 0..3 of bm
#define RED(i) vcgmax(cg##i, a##i, pAll); vdintlv(ev##i, od##i, cg##i, cg##i);              \
            vmax(bm##i, ev##i, od##i, pAll);                                                \
            vshrs(bd##i,(vector_u16&)bm##i,(int16_t)10,pAll,MODE_ZEROING);                  \
            vand(bd##i,bd##i,c1F,pAll,MODE_ZEROING);   /* biased5 in lanes 0..3 */
            DOU8(RED)
#undef RED
            // reciprocal multiplier field (mf, lanes 0..3), then e8m0 store (reuse bd)
#define MF(i) vsub(mf##i,c32,bd##i,pAll); vmaxs(mf##i,mf##i,(uint16_t)1,pAll);              \
            vmins(mf##i,mf##i,(uint16_t)30,pAll); vshls(mf##i,mf##i,(int16_t)10,pAll,MODE_ZEROING); \
            vadds(bd##i,bd##i,(uint16_t)110,pAll);                                          \
            vsts(bd##i,(__ubuf__ uint16_t*)(sb_s+(uint32_t)(row+i)*SCALE_STRIDE),0,NORM_B16,p4);
            DOU8(MF)
#undef MF
            // broadcast mf lanes 0..3 -> [m0x32,m1x32,m2x32,m3x32] (5x upsample)
#define BC(i) vintlv(lo##i, hi, mf##i, mf##i); vintlv(lo##i, hi, lo##i, lo##i);             \
            vintlv(lo##i, hi, lo##i, lo##i); vintlv(lo##i, hi, lo##i, lo##i);               \
            vintlv(lo##i, hi, lo##i, lo##i);                                                \
            vmul(v##i, v##i, (vector_f16&)lo##i, pAll, MODE_ZEROING);
            DOU8(BC)
#undef BC
            // f16 -> bf16 -> fp4, pack (PK4_B32), store 64 B/row
#define CS(i) vcvt(cb##i, v##i, pAll, ROUND_R); vcvt(q##i, cb##i, pAll, ROUND_R, PART_P0);  \
            vsts((vector_u8&)q##i,(__ubuf__ uint8_t*)(sb_q+(uint32_t)(row+i)*(HAD_N/2)),0,PK4_B32,pq);
            DOU8(CS)
#undef CS
        }
    }
}
#undef DOU8
#undef DOU4
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

__global__ AICORE void fused_hadamard_mxfp4_a5(__gm__ void *x_gm, __gm__ void *q_gm,
                                               __gm__ void *s_gm, uint32_t batch)
{
#ifdef __DAV_VEC__
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_ctrl(static_cast<uint64_t>(1) << 50);   // clip into MAX_NORM on cast

    using XSh = pto::Shape<1, 1, 1, 1, FLAT_IN>;
    using XSt = pto::Stride<1, 1, 1, FLAT_IN, 1>;
    using XTile = Tile<TileType::Vec, in_t, 1, FLAT_IN, BLayout::RowMajor, 1, FLAT_IN>;
    using QSh = pto::Shape<1, 1, 1, 1, Q_BYTES>;
    using QSt = pto::Stride<1, 1, 1, Q_BYTES, 1>;
    using QTile = Tile<TileType::Vec, uint8_t, 1, Q_BYTES, BLayout::RowMajor, 1, Q_BYTES>;
    using SSh = pto::Shape<1, 1, 1, 1, S_BYTES>;
    using SSt = pto::Stride<1, 1, 1, S_BYTES, 1>;
    using STile = Tile<TileType::Vec, uint8_t, 1, S_BYTES, BLayout::RowMajor, 1, S_BYTES>;

    const unsigned xo[2] = {X0, X1};
    const unsigned qo[2] = {Q0, Q1};
    const unsigned so[2] = {S0, S1};
    const event_t ev[2] = {EVENT_ID0, EVENT_ID1};

    const uint32_t cid = get_block_idx();
    const uint32_t num_cores = get_block_num();
    const uint32_t tiles = batch / ROWS_PER_TILE;

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    uint32_t it = 0;
    for (uint32_t tb = cid; tb < tiles; tb += num_cores, ++it) {
        const int pp = it & 1;
        const uint64_t xoff = static_cast<uint64_t>(tb) * FLAT_IN;
        const uint64_t qoff = static_cast<uint64_t>(tb) * Q_BYTES;
        const uint64_t soff = static_cast<uint64_t>(tb) * S_BYTES;

        wait_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
        XTile xt;  TASSIGN(xt, xo[pp]);
        GlobalTensor<in_t, XSh, XSt> gx(reinterpret_cast<__gm__ in_t *>(x_gm) + xoff, XSh());
        TLOAD(xt, gx);
        set_flag(PIPE_MTE2, PIPE_V, ev[pp]);
        wait_flag(PIPE_MTE2, PIPE_V, ev[pp]);

#ifndef PHASE_SEL
#define PHASE_SEL 0     // 0=both, 1=butterfly only, 2=quant only (perf ablation)
#endif
#if PHASE_SEL != 2
        hadamard_phase((__ubuf__ in_t *)(uintptr_t)xo[pp],
                       (__ubuf__ half *)(uintptr_t)W0, ROWS_PER_TILE);
#endif
#if PHASE_SEL != 1
        quant_phase((__ubuf__ half *)(uintptr_t)W0,
                    (__ubuf__ uint8_t *)(uintptr_t)qo[pp],
                    (__ubuf__ uint8_t *)(uintptr_t)so[pp], ROWS_PER_TILE);
#endif

        set_flag(PIPE_V, PIPE_MTE3, ev[pp]);
        wait_flag(PIPE_V, PIPE_MTE3, ev[pp]);
        QTile qt;  TASSIGN(qt, qo[pp]);
        STile st;  TASSIGN(st, so[pp]);
        GlobalTensor<uint8_t, QSh, QSt> gq(reinterpret_cast<__gm__ uint8_t *>(q_gm) + qoff, QSh());
        GlobalTensor<uint8_t, SSh, SSt> gs(reinterpret_cast<__gm__ uint8_t *>(s_gm) + soff, SSh());
        TSTORE(gq, qt);
        TSTORE(gs, st);
        set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#else
    (void)x_gm; (void)q_gm; (void)s_gm; (void)batch;
#endif
}

extern "C" void call_fused_hadamard_mxfp4_a5(uint32_t block_dim, void *stream,
                                             uint8_t *x, uint8_t *q, uint8_t *s, uint32_t batch)
{
    fused_hadamard_mxfp4_a5<<<block_dim, nullptr, stream>>>(x, q, s, batch);
}
