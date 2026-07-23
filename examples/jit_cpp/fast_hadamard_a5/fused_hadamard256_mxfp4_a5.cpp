// fused_hadamard256_mxfp4_a5 — N=256 Walsh-Hadamard (deinterleave-LOAD butterfly)
// fused with MXFP4 (e2m1, block=32) quant, on the A5 vector core.
//
// Butterfly: each of the 8 stages does the even/odd split on the MTE2 load
// (vlds DINTLV_B16) and the concat-halves recombine on the MTE3 store (vsts to
// [0:128]/[128:256]) -- memory-bound, ~copy speed. Quant: the register-resident
// vcgmax path, applied to each 128-half of the 256-row (8 blocks of 32).
//
// Build: bisheng --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE
//        -DHAD_IN_BF16={0,1} [-DROWS_PER_TILE=<rows>]
#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef HAD_N
#define HAD_N 256
#endif
#ifndef HAD_LOG2N
#define HAD_LOG2N 8
#endif
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 64
#endif
#ifndef HAD_IN_BF16
#define HAD_IN_BF16 0
#endif
#define MX_BLK 32
constexpr float HAD_INV = 0.0625f;    // 1/sqrt(256)

#ifdef __CCE_AICORE__
#if HAD_IN_BF16
using in_t = bfloat16_t;
#else
using in_t = half;
#endif
constexpr unsigned LANES = 128;
constexpr unsigned SUBROWS = ROWS_PER_TILE * (HAD_N / LANES);   // 128-wide subrows
constexpr unsigned SCALE_STRIDE = 32;                            // bytes / 128-subrow (padded)
constexpr unsigned FLAT_IN = ROWS_PER_TILE * HAD_N;
constexpr unsigned X_BYTES = FLAT_IN * sizeof(in_t);
constexpr unsigned W_BYTES = FLAT_IN * sizeof(half);
constexpr unsigned Q_BYTES = SUBROWS * (LANES / 2);              // 64 fp4 bytes/subrow
constexpr unsigned S_BYTES = SUBROWS * SCALE_STRIDE;
constexpr unsigned aln(unsigned b){ return (b+511u)&~511u; }
constexpr unsigned Q0 = 0;
constexpr unsigned S0 = Q0 + aln(Q_BYTES);
constexpr unsigned W0 = S0 + aln(S_BYTES);
constexpr unsigned Q1 = W0 + aln(W_BYTES);
constexpr unsigned S1 = Q1 + aln(Q_BYTES);
constexpr unsigned W1 = S1 + aln(S_BYTES);
constexpr unsigned UB_END = W1 + aln(W_BYTES);
static_assert(UB_END <= 192u*1024u, "UB overflow");

#ifdef __DAV_VEC__
#define DOU8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define DOU4(M) M(0) M(1) M(2) M(3)

// Deinterleave-load 256-point WHT, in-place on wb (f16). If bf16 input, wb was
// filled from bf16; we widen on the first load only... simpler: caller loads X
// (in_t) into wb reinterpreted; for bf16 we convert during first stage.
__tf__ static AICORE void bfly256(__ubuf__ half *wb, uint32_t rows)
{
    __VEC_SCOPE__
    {
        uint32_t la = LANES; MaskReg pAll = CreatePredicate<half>(la);
        vector_f16 e0,e1,e2,e3,e4,e5,e6,e7, o0,o1,o2,o3,o4,o5,o6,o7;
        vector_f16 s0,s1,s2,s3,s4,s5,s6,s7, d0,d1,d2,d3,d4,d5,d6,d7;
#if HAD_IN_BF16
        // input arrived as bf16 in wb; widen to f16 in place before the butterfly
        {
            vector_bf16 wbf; vector_f16 wf;
            uint32_t nchunk = (uint32_t)rows * (HAD_N / LANES);
            for (uint16_t c = 0; c < (uint16_t)nchunk; ++c) {
                vlds(wbf, (__ubuf__ bfloat16_t *)wb, (uint32_t)c * LANES, NORM);
                vcvt(wf, wbf, pAll, RS_DISABLE, ROUND_R);
                vsts(wf, wb, (uint32_t)c * LANES, NORM_B16, pAll);
            }
            mem_bar(VST_VLD);
        }
#endif
        for (uint16_t st = 0; st < (uint16_t)HAD_LOG2N; ++st) {
            for (uint16_t row = 0; row < (uint16_t)rows; row += 8) {
                const uint32_t base = (uint32_t)row * HAD_N;
#define LD(i) vlds(e##i, o##i, wb + base + (uint32_t)(i)*HAD_N, 0, DINTLV_B16);
                DOU8(LD)
#undef LD
#define AD(i) vadd(s##i, e##i, o##i, pAll);
                DOU8(AD)
#undef AD
#define SU(i) vsub(d##i, e##i, o##i, pAll);
                DOU8(SU)
#undef SU
#define ST(i) vsts(s##i, wb + base + (uint32_t)(i)*HAD_N,        0, NORM_B16, pAll); \
              vsts(d##i, wb + base + (uint32_t)(i)*HAD_N + LANES, 0, NORM_B16, pAll);
                DOU8(ST)
#undef ST
            }
            mem_bar(VST_VLD);
        }
    }
}

// MXFP4 quant of the WHT result, per 128-wide subrow (4 blocks of 32).
// Applies the 1/sqrt(256) scale on load. subrows = ROWS_PER_TILE * 2.
__tf__ static AICORE void quant256(__ubuf__ half *wb, __ubuf__ uint8_t *sb_q,
                                   __ubuf__ uint8_t *sb_s, uint32_t subrows)
{
    __VEC_SCOPE__
    {
        uint32_t la = LANES; MaskReg pAll = CreatePredicate<half>(la);
        uint32_t f4 = 4; MaskReg p4 = CreatePredicate<half>(f4);
        uint32_t s64 = 64; MaskReg pq = CreatePredicate<float>(s64);
        vector_u16 c1F, c32; vbr(c1F, (uint16_t)0x1F); vbr(c32, (uint16_t)32);
        vector_f16 v0,v1,v2,v3, a0,a1,a2,a3, cg0,cg1,cg2,cg3, ev0,ev1,ev2,ev3;
        vector_f16 od0,od1,od2,od3, bm0,bm1,bm2,bm3;
        vector_u16 bd0,bd1,bd2,bd3, mf0,mf1,mf2,mf3, lo0,lo1,lo2,lo3, hi;
        vector_bf16 cb0,cb1,cb2,cb3;
        vector_f4e2m1x2 q0,q1,q2,q3;
        for (uint16_t r = 0; r < (uint16_t)subrows; r += 4) {
#define LDW(i) vlds(v##i, wb, (uint32_t)(r+i)*LANES, NORM); vmuls(v##i,v##i,(half)HAD_INV,pAll,MODE_ZEROING); \
            vabs(a##i, v##i, pAll);
            DOU4(LDW)
#undef LDW
#define RED(i) vcgmax(cg##i, a##i, pAll); vdintlv(ev##i, od##i, cg##i, cg##i);              \
            vmax(bm##i, ev##i, od##i, pAll);                                                \
            vshrs(bd##i,(vector_u16&)bm##i,(int16_t)10,pAll,MODE_ZEROING);                  \
            vand(bd##i,bd##i,c1F,pAll,MODE_ZEROING);
            DOU4(RED)
#undef RED
#define MF(i) vsub(mf##i,c32,bd##i,pAll); vmaxs(mf##i,mf##i,(uint16_t)1,pAll);              \
            vmins(mf##i,mf##i,(uint16_t)30,pAll); vshls(mf##i,mf##i,(int16_t)10,pAll,MODE_ZEROING); \
            vadds(bd##i,bd##i,(uint16_t)110,pAll);                                          \
            vsts(bd##i,(__ubuf__ uint16_t*)(sb_s+(uint32_t)(r+i)*SCALE_STRIDE),0,NORM_B16,p4);
            DOU4(MF)
#undef MF
#define BC(i) vintlv(lo##i, hi, mf##i, mf##i); vintlv(lo##i, hi, lo##i, lo##i);             \
            vintlv(lo##i, hi, lo##i, lo##i); vintlv(lo##i, hi, lo##i, lo##i);               \
            vintlv(lo##i, hi, lo##i, lo##i);                                                \
            vmul(v##i, v##i, (vector_f16&)lo##i, pAll, MODE_ZEROING);
            DOU4(BC)
#undef BC
#define CS(i) vcvt(cb##i, v##i, pAll, ROUND_R); vcvt(q##i, cb##i, pAll, ROUND_R, PART_P0);  \
            vsts((vector_u8&)q##i,(__ubuf__ uint8_t*)(sb_q+(uint32_t)(r+i)*(LANES/2)),0,PK4_B32,pq);
            DOU4(CS)
#undef CS
        }
    }
}
#undef DOU8
#undef DOU4
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

__global__ AICORE void fused_hadamard256_mxfp4_a5(__gm__ void *x_gm, __gm__ void *q_gm,
                                                  __gm__ void *s_gm, uint32_t batch)
{
#ifdef __DAV_VEC__
    set_mask_norm(); set_vector_mask(-1, -1);
    set_ctrl(static_cast<uint64_t>(1) << 50);
    using XSh = pto::Shape<1,1,1,1,FLAT_IN>; using XSt = pto::Stride<1,1,1,FLAT_IN,1>;
    using XTile = Tile<TileType::Vec, in_t, 1, FLAT_IN, BLayout::RowMajor, 1, FLAT_IN>;
    using QSh = pto::Shape<1,1,1,1,Q_BYTES>; using QSt = pto::Stride<1,1,1,Q_BYTES,1>;
    using QTile = Tile<TileType::Vec, uint8_t, 1, Q_BYTES, BLayout::RowMajor, 1, Q_BYTES>;
    using SSh = pto::Shape<1,1,1,1,S_BYTES>; using SSt = pto::Stride<1,1,1,S_BYTES,1>;
    using STile = Tile<TileType::Vec, uint8_t, 1, S_BYTES, BLayout::RowMajor, 1, S_BYTES>;
    const unsigned wo[2] = {W0, W1}, qo[2] = {Q0, Q1}, so[2] = {S0, S1};
    const event_t ev[2] = {EVENT_ID0, EVENT_ID1};
    const uint32_t cid = get_block_idx(), nc = get_block_num();
    const uint32_t tiles = batch / ROWS_PER_TILE;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    uint32_t it = 0;
    for (uint32_t tb = cid; tb < tiles; tb += nc, ++it) {
        const int pp = it & 1;
        const uint64_t xoff = (uint64_t)tb * FLAT_IN, qoff = (uint64_t)tb * Q_BYTES, soff = (uint64_t)tb * S_BYTES;
        wait_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
        XTile xt; TASSIGN(xt, wo[pp]);   // load X into the W buffer
        GlobalTensor<in_t, XSh, XSt> gx((__gm__ in_t*)x_gm + xoff, XSh());
        TLOAD(xt, gx);
        set_flag(PIPE_MTE2, PIPE_V, ev[pp]); wait_flag(PIPE_MTE2, PIPE_V, ev[pp]);
        bfly256((__ubuf__ half*)(uintptr_t)wo[pp], ROWS_PER_TILE);
        quant256((__ubuf__ half*)(uintptr_t)wo[pp],
                 (__ubuf__ uint8_t*)(uintptr_t)qo[pp],
                 (__ubuf__ uint8_t*)(uintptr_t)so[pp], SUBROWS);
        set_flag(PIPE_V, PIPE_MTE3, ev[pp]); wait_flag(PIPE_V, PIPE_MTE3, ev[pp]);
        QTile qt; TASSIGN(qt, qo[pp]); STile st; TASSIGN(st, so[pp]);
        GlobalTensor<uint8_t, QSh, QSt> gq((__gm__ uint8_t*)q_gm + qoff, QSh());
        GlobalTensor<uint8_t, SSh, SSt> gs((__gm__ uint8_t*)s_gm + soff, SSh());
        TSTORE(gq, qt); TSTORE(gs, st);
        set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#else
    (void)x_gm;(void)q_gm;(void)s_gm;(void)batch;
#endif
}

extern "C" void call_fused_hadamard256_mxfp4_a5(uint32_t bd, void *s, uint8_t *x,
                                                uint8_t *q, uint8_t *sc, uint32_t b)
{ fused_hadamard256_mxfp4_a5<<<bd, nullptr, s>>>(x, q, sc, b); }
