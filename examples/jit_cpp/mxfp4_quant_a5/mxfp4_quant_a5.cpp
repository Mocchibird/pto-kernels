// mxfp4_quant_a5 — OCP MXFP4 quantization on the Ascend A5 (dav-c310) vector
// core.
//
//   in   (batch, K) bf16
//   out  (batch, K/2)  uint8   packed E2M1 nibbles, element 0 in the LOW nibble
//        (batch, K/32) uint8   E8M0 per-32-element block scales
//
// Every convention below is device-measured, not assumed (see PLAN.md section
// 0):
//   * scale rule is OCP Algorithm 1 FLOOR: byte = floor(log2 amax) - 2 + 127,
//   which
//     for bf16 is exactly `b - 2` where b is amax's biased exponent field. So
//     the scale is pure integer arithmetic on the exponent -- no logs, no
//     division.
//   * the reciprocal multiplier 1/X is the bf16 whose exponent field is 256 - b
//   and
//     whose mantissa is 0. CLAMP b, never the multiplier: deriving both from
//     the same clamped b keeps them exact inverses (the WIP clamped the
//     multiplier and silently dequantized 2x/4x low on near-zero blocks).
//   * the bf16 -> fp4 cast SATURATES at +/-6.0, so the FLOOR rule's amax/X up
//   to 7.0
//     needs no clipping and no set_ctrl.
//   * PART_Px deposits its 64 output codes at byte stride 4, offset x, so part
//   x must
//     be fed elements 8k+2x and 8k+2x+1 -- not four sequential blocks.
#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef QK
#define QK 32  // MXFP4 block: 32 elements share one E8M0 scale
#endif
#ifndef KWIDTH
#define KWIDTH 256  // elements per row; must be a multiple of 256
#endif
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 16
#endif
#ifndef NBUF
#define NBUF 4
#endif
#ifndef PREFETCH
#define PREFETCH 2
#endif
#ifndef UB_USABLE_BYTES
#define UB_USABLE_BYTES (192u * 1024u)
#endif

#ifdef __CCE_AICORE__
constexpr unsigned VL_B16 = 128;                // bf16 lanes per vector
constexpr unsigned BLKS_PER_ROW = KWIDTH / QK;  // scale bytes per row
constexpr unsigned IN_ELEMS = ROWS_PER_TILE * KWIDTH;
constexpr unsigned Q_BYTES = IN_ELEMS / 2;  // packed fp4
constexpr unsigned S_BYTES = ROWS_PER_TILE * BLKS_PER_ROW;
constexpr unsigned IN_BYTES = IN_ELEMS * 2;

#define ALN(b) ((((unsigned)(b)) + 511u) & ~511u)
// UB layout: NBUF input tiles, then one q and one s staging buffer per input
// tile.
#define XOFF(i) ((unsigned)(i) * ALN(IN_BYTES))
#define QOFF(i) (NBUF * ALN(IN_BYTES) + (unsigned)(i) * ALN(Q_BYTES))
#define SOFF(i) \
  (NBUF * (ALN(IN_BYTES) + ALN(Q_BYTES)) + (unsigned)(i) * ALN(S_BYTES))
constexpr unsigned UB_NEEDED =
    NBUF * (ALN(IN_BYTES) + ALN(Q_BYTES) + ALN(S_BYTES));

static_assert(KWIDTH % 256 == 0, "KWIDTH must be a multiple of 256");
static_assert(QK == 32, "MXFP4 block size is 32 by definition");
static_assert(UB_NEEDED <= UB_USABLE_BYTES, "UB overflow");
static_assert(PREFETCH < NBUF,
              "PREFETCH must be < NBUF (else the pipeline deadlocks)");

// bf16 bit layout: [sign 15][exp 14:7][mant 6:0], bias 127.
constexpr uint16_t BF16_ABS =
    0x7FFFu;                         // clear the sign for a magnitude compare
constexpr uint16_t E8M0_OFF = 2u;    // byte = b - 2  (= b + 127-2-127)
constexpr uint16_t MULT_OFF = 256u;  // multiplier exponent field = 256 - b
constexpr uint16_t B_MIN = 2u;       // keeps 1/X finite and normal
constexpr uint16_t B_MAX = 254u;
#endif

#ifdef __DAV_VEC__
// One 256-element window: 2 bf16 registers in, 128 packed bytes out, 8 scale
// bytes.
__tf__ static AICORE void quant_window(__ubuf__ bfloat16_t *xb,
                                       __ubuf__ uint8_t *qb,
                                       __ubuf__ uint8_t *sb, uint32_t windows) {
  __VEC_SCOPE__ {
    uint32_t vl = VL_B16;
    uint32_t vl64 = 64;
    uint32_t vl128 = 128;
    MaskReg pAll = CreatePredicate<bfloat16_t>(vl);
    MaskReg p64 = CreatePredicate<bfloat16_t>(vl64);  // one cast eats 64 lanes
    MaskReg pB8 = CreatePredicate<uint8_t>(vl128);    // 128 packed bytes out

    vector_bf16 v0, m0;
    vector_u16 a0, e0, mu0, c256, cabs, clow;
    vector_f4e2m1x2 q;
    // hoisted constants: there is no scalar-immediate bitwise AND on this ISA
    vdup(c256, MULT_OFF, pAll, MODE_ZEROING);
    vdup(cabs, BF16_ABS, pAll, MODE_ZEROING);
    vdup(clow, (uint16_t)0x00FFu, pAll, MODE_ZEROING);

    // One iteration = one 256-element window = four 64-lane casts into the four
    // byte-phases of a single f4e2m1x2 register (PART_Px writes bytes 4j+x, and
    // one cast consumes 64 lanes -- both measured). The resulting register is
    // byte-interleaved across the phases; the test reports the exact
    // permutation.
    for (uint16_t w = 0; w < (uint16_t)windows; ++w) {
      const uint32_t xo = (uint32_t)w * 256u;
      const uint32_t qo = (uint32_t)w * 128u;
      const uint32_t so = (uint32_t)w * 8u;

#define QUARTER(IDX, PARTSEL)                          \
  vlds(v0, xb + xo + (IDX) * 64u, 0, NORM);            \
  vand(a0, (vector_u16 &)v0, cabs, p64);               \
  vcgmax(a0, a0, p64);                                 \
  vshrs(e0, a0, (int16_t)7, p64, MODE_ZEROING);        \
  vand(e0, e0, clow, p64);                             \
  vmaxs(e0, e0, B_MIN, p64);                           \
  vmins(e0, e0, B_MAX, p64);                           \
  vsub(mu0, c256, e0, p64);                            \
  vshls(mu0, mu0, (int16_t)7, p64, MODE_ZEROING);      \
  vmul(m0, v0, (vector_bf16 &)mu0, p64);               \
  vcvt(q, m0, p64, ROUND_R, PARTSEL);                  \
  vadds(e0, e0, (uint16_t)(0x10000u - E8M0_OFF), p64); \
  vsts((vector_u8 &)e0, sb + so + (IDX) * 2u, 0, ONEPT_B8, p64);

      QUARTER(0u, PART_P0)
      QUARTER(1u, PART_P1)
      QUARTER(2u, PART_P2)
      QUARTER(3u, PART_P3)
#undef QUARTER
      vsts((vector_u8 &)q, qb + qo, 0, NORM_B8, pB8);
    }
    mem_bar(VST_VLD);
  }
}
#endif  // __DAV_VEC__

__global__ AICORE void mxfp4_quant(__gm__ void *x_gm, __gm__ void *q_gm,
                                   __gm__ void *s_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  set_mask_norm();
  set_vector_mask(-1, -1);

  using ShX = pto::Shape<1, 1, 1, 1, IN_ELEMS>;
  using StX = pto::Stride<1, 1, 1, IN_ELEMS, 1>;
  using TX = Tile<TileType::Vec, bfloat16_t, 1, IN_ELEMS, BLayout::RowMajor, 1,
                  IN_ELEMS>;
  using ShQ = pto::Shape<1, 1, 1, 1, Q_BYTES>;
  using StQ = pto::Stride<1, 1, 1, Q_BYTES, 1>;
  using TQ =
      Tile<TileType::Vec, uint8_t, 1, Q_BYTES, BLayout::RowMajor, 1, Q_BYTES>;
  using ShS = pto::Shape<1, 1, 1, 1, S_BYTES>;
  using StS = pto::Stride<1, 1, 1, S_BYTES, 1>;
  using TS =
      Tile<TileType::Vec, uint8_t, 1, S_BYTES, BLayout::RowMajor, 1, S_BYTES>;

  const uint32_t cid = get_block_idx(), nc = get_block_num();
  const uint32_t tiles = batch / ROWS_PER_TILE;
  const event_t ev[8] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
                         EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7};

#define ISSUE_LOAD(K)                                                   \
  do {                                                                  \
    uint32_t _tb = cid + (uint32_t)(K) * nc;                            \
    if (_tb < tiles) {                                                  \
      const int _pp = (uint32_t)(K) % NBUF;                             \
      wait_flag(PIPE_MTE3, PIPE_MTE2, ev[_pp]);                         \
      TX _xt;                                                           \
      TASSIGN(_xt, XOFF(_pp));                                          \
      GlobalTensor<bfloat16_t, ShX, StX> _g(                            \
          (__gm__ bfloat16_t *)x_gm + (uint64_t)_tb * IN_ELEMS, ShX()); \
      TLOAD(_xt, _g);                                                   \
      set_flag(PIPE_MTE2, PIPE_V, ev[_pp]);                             \
    }                                                                   \
  } while (0)

  for (int i = 0; i < NBUF; ++i) set_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
  for (uint32_t kk = 0; kk < (uint32_t)PREFETCH; ++kk) ISSUE_LOAD(kk);

  uint32_t k = 0;
  for (uint32_t tb = cid; tb < tiles; tb += nc, ++k) {
    const int pp = k % NBUF;
    ISSUE_LOAD(k + PREFETCH);
    wait_flag(PIPE_MTE2, PIPE_V, ev[pp]);

    quant_window((__ubuf__ bfloat16_t *)(uintptr_t)XOFF(pp),
                 (__ubuf__ uint8_t *)(uintptr_t)QOFF(pp),
                 (__ubuf__ uint8_t *)(uintptr_t)SOFF(pp), IN_ELEMS / 256u);

    set_flag(PIPE_V, PIPE_MTE3, ev[pp]);
    wait_flag(PIPE_V, PIPE_MTE3, ev[pp]);
    TQ qt;
    TASSIGN(qt, QOFF(pp));
    GlobalTensor<uint8_t, ShQ, StQ> gq(
        (__gm__ uint8_t *)q_gm + (uint64_t)tb * Q_BYTES, ShQ());
    TSTORE(gq, qt);
    TS st;
    TASSIGN(st, SOFF(pp));
    GlobalTensor<uint8_t, ShS, StS> gs(
        (__gm__ uint8_t *)s_gm + (uint64_t)tb * S_BYTES, ShS());
    TSTORE(gs, st);
    set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
  }
  for (int i = 0; i < NBUF; ++i) wait_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
#else
  (void)x_gm;
  (void)q_gm;
  (void)s_gm;
  (void)batch;
#endif
}

extern "C" void call_mxfp4_quant(uint32_t bd, void *s, uint8_t *x, uint8_t *q,
                                 uint8_t *sc, uint32_t batch) {
  mxfp4_quant<<<bd, nullptr, s>>>(x, q, sc, batch);
}
