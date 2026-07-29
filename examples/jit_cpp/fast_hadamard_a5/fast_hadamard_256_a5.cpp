// fast_hadamard_256_a5 — N=256 Walsh-Hadamard via DEINTERLEAVE-LOAD butterfly.
// Each stage does the even/odd split on the MTE2 load (vlds DINTLV_B16) and the
// concat-halves recombine on the MTE3 store (vsts to [0:128] / [128:256]),
// leaving only vadd/vsub on the vector-execute pipe. In-place on a UB tile.
// Compares against a plain copy-floor of the same tiling.
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
#ifndef HAD_UNROLL
#define HAD_UNROLL 8
#endif

#ifdef __CCE_AICORE__
constexpr unsigned LANES = 128;
constexpr unsigned FLAT = ROWS_PER_TILE * HAD_N;          // f16 elems/tile
constexpr unsigned X_BYTES = FLAT * sizeof(half);
constexpr unsigned aln(unsigned b){ return (b+511u)&~511u; }
#ifndef NBUF
#define NBUF 4                       // pipeline depth (buffers)
#endif
#ifndef PREFETCH
#define PREFETCH 2                   // tiles to prefetch ahead
#endif
#define XOFF(i) ((unsigned)(i) * ((X_BYTES + 511u) & ~511u))
// A5 UB is 248 KB physical, but this kernel's per-buffer event-ID reuse tops out
// at ~NBUF=4 (deeper -> runtime device fault 507035), which fits 192 KB for these
// tiles -- so UB capacity is not the bottleneck here (see bench256_nbuf.py).
#ifndef UB_USABLE_BYTES
#define UB_USABLE_BYTES (192u * 1024u)
#endif
static_assert(NBUF * aln(X_BYTES) <= UB_USABLE_BYTES, "UB overflow");

#ifdef __DAV_VEC__
#define DOU(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
// Deinterleave-LOAD 256-point WHT: even/odd split on the MTE load (vlds
// DINTLV_B16), concat-halves recombine on the store (vsts to [0:128]/[128:256]),
// only vadd/vsub on the vector-execute pipe -> memory-bound (~copy speed).
__tf__ static AICORE void bfly256(__ubuf__ half *wb, uint32_t rows)
{
    __VEC_SCOPE__
    {
        uint32_t la = LANES; MaskReg pAll = CreatePredicate<half>(la);
        vector_f16 e0,e1,e2,e3,e4,e5,e6,e7, o0,o1,o2,o3,o4,o5,o6,o7;
        vector_f16 s0,s1,s2,s3,s4,s5,s6,s7, d0,d1,d2,d3,d4,d5,d6,d7;
        for (uint16_t st = 0; st < (uint16_t)HAD_LOG2N; ++st) {
            for (uint16_t row = 0; row < (uint16_t)rows; row += HAD_UNROLL) {
                const uint32_t base = (uint32_t)row * HAD_N;
#define LD(i) vlds(e##i, o##i, wb + base + (uint32_t)(i)*HAD_N, 0, DINTLV_B16);
                DOU(LD)
#undef LD
#define AD(i) vadd(s##i, e##i, o##i, pAll);
                DOU(AD)
#undef AD
#define SU(i) vsub(d##i, e##i, o##i, pAll);
                DOU(SU)
#undef SU
#define ST(i) vsts(s##i, wb + base + (uint32_t)(i)*HAD_N,       0, NORM_B16, pAll); \
              vsts(d##i, wb + base + (uint32_t)(i)*HAD_N + LANES, 0, NORM_B16, pAll);
                DOU(ST)
#undef ST
            }
            mem_bar(VST_VLD);
        }
    }
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

__global__ AICORE void hadamard256(__gm__ void *x_gm, uint32_t batch)
{
#ifdef __DAV_VEC__
    set_mask_norm(); set_vector_mask(-1, -1);
    using Sh = pto::Shape<1,1,1,1,FLAT>; using St = pto::Stride<1,1,1,FLAT,1>;
    using T = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;
    const event_t ev[8] = {EVENT_ID0,EVENT_ID1,EVENT_ID2,EVENT_ID3,EVENT_ID4,EVENT_ID5,EVENT_ID6,EVENT_ID7};
    const unsigned xoff[4] = {XOFF(0), XOFF(1), XOFF(2), XOFF(3)};
    const uint32_t cid = get_block_idx(), nc = get_block_num();
    const uint32_t tiles = batch / ROWS_PER_TILE;

    // issue the async load for this core's K-th tile into buffer K%NBUF
#define ISSUE_LOAD(K) do {                                                     \
        uint32_t _tb = cid + (uint32_t)(K) * nc;                               \
        if (_tb < tiles) {                                                     \
            const int _pp = (uint32_t)(K) % NBUF;                              \
            wait_flag(PIPE_MTE3, PIPE_MTE2, ev[_pp]);                          \
            T _xt; TASSIGN(_xt, xoff[_pp]);                                    \
            GlobalTensor<half, Sh, St> _g((__gm__ half*)x_gm + (uint64_t)_tb * FLAT, Sh()); \
            TLOAD(_xt, _g);                                                    \
            set_flag(PIPE_MTE2, PIPE_V, ev[_pp]);                              \
        } } while (0)

    for (int i = 0; i < NBUF; ++i) set_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);  // all free
    for (uint32_t kk = 0; kk < (uint32_t)PREFETCH; ++kk) ISSUE_LOAD(kk);   // prologue

    uint32_t k = 0;
    for (uint32_t tb = cid; tb < tiles; tb += nc, ++k) {
        const int pp = k % NBUF;
        ISSUE_LOAD(k + PREFETCH);                    // prefetch (overlaps this compute)
        wait_flag(PIPE_MTE2, PIPE_V, ev[pp]);        // this tile's load done
        bfly256((__ubuf__ half*)(uintptr_t)xoff[pp], ROWS_PER_TILE);
        set_flag(PIPE_V, PIPE_MTE3, ev[pp]); wait_flag(PIPE_V, PIPE_MTE3, ev[pp]);
        T xt; TASSIGN(xt, XOFF(pp));
        GlobalTensor<half, Sh, St> g((__gm__ half*)x_gm + (uint64_t)tb * FLAT, Sh());
        TSTORE(g, xt);
        set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);      // buffer free
    }
    for (int i = 0; i < NBUF; ++i) wait_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
#else
    (void)x_gm;(void)batch;
#endif
}

__global__ AICORE void copy256(__gm__ void *x_gm, uint32_t batch)
{
#ifdef __DAV_VEC__
    using Sh = pto::Shape<1,1,1,1,FLAT>; using St = pto::Stride<1,1,1,FLAT,1>;
    using T = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;
    const unsigned xo[2] = {XOFF(0), XOFF(1)}; const event_t ev[2] = {EVENT_ID0, EVENT_ID1};
    const uint32_t cid = get_block_idx(), nc = get_block_num();
    const uint32_t tiles = batch / ROWS_PER_TILE;
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    uint32_t it = 0;
    for (uint32_t tb = cid; tb < tiles; tb += nc, ++it) {
        const int pp = it & 1; const uint64_t off = (uint64_t)tb * FLAT;
        wait_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
        T xt; TASSIGN(xt, xo[pp]);
        GlobalTensor<half, Sh, St> g((__gm__ half*)x_gm + off, Sh());
        TLOAD(xt, g);
        set_flag(PIPE_MTE2, PIPE_MTE3, ev[pp]); wait_flag(PIPE_MTE2, PIPE_MTE3, ev[pp]);
        TSTORE(g, xt);
        set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#else
    (void)x_gm;(void)batch;
#endif
}

extern "C" void call_hadamard256(uint32_t bd, void *s, uint8_t *x, uint32_t b)
{ hadamard256<<<bd, nullptr, s>>>(x, b); }
extern "C" void call_copy256(uint32_t bd, void *s, uint8_t *x, uint32_t b)
{ copy256<<<bd, nullptr, s>>>(x, b); }
