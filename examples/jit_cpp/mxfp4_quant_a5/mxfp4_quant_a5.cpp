// mxfp4_quant_a5 — MXFP4 block quantization on Ascend A5 (dav-c310). A5 only.
//
// (batch, K) bf16 -> q (batch, K/2) uint8 + scale (batch, K/32) uint8. 32
// elements share one E8M0 byte; each element becomes one E2M1 nibble. OCP MX
// v1.0 6.3 Algorithm 1 (FLOOR): byte = b - 2, 1/X exponent field = 256 - b,
// same clamped b.
#include <pto/pto-inst.hpp>
#include <utility>
using namespace pto;

// Row widths with an instantiation. Adding one here is the only edit needed.
constexpr unsigned SUPPORTED_K[] = {128, 256, 512, 1024, 2048, 4096};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_K) / sizeof(SUPPORTED_K[0]);
constexpr unsigned DEFAULT_K = 4096;  // used when a caller does not choose
constexpr unsigned MX_BLOCK = 32;    // MXFP4 block: 32 elements, one E8M0 scale
constexpr unsigned DEF_BUFFERS = 4;  // UB buffers in the GM<->UB pipeline
constexpr unsigned DEF_PREFETCH = 2;  // tiles in flight ahead of the compute

// 32 KB bf16 tile. Must match rows_for() in jit_util_mxfp4_a5.py. Measured
// against 8192 and 32768: 8192 leaves DMA on the table (its own no-compute
// floor is 6-11% below this one's), and 32768 only fits with NBuffers=3 and
// then reaches 98% of UB.
constexpr unsigned TILE_ELEMS = 16384;
constexpr unsigned PASS_B_GRAIN = 4096;  // 128 blocks of 32
template <unsigned K>
struct RowsFor {
  static constexpr unsigned quotient = TILE_ELEMS / K;
  static constexpr unsigned value = quotient > 1u ? quotient : 1u;
};

#ifdef __CCE_AICORE__
constexpr unsigned B16_LANES = 128;  // bf16 lanes in one vector register
// MEASURED: vcgmax on b16 groups 16 lanes, 8 results in lanes 0..7.
constexpr unsigned VCGMAX_B16_GROUP = 16;
constexpr unsigned VCGMAX_B16_RESULTS = B16_LANES / VCGMAX_B16_GROUP;
static_assert(VCGMAX_B16_RESULTS == 8, "abs_block_max stores with PAT_VL8");
// RULE: vsts needs a 32-byte-aligned UB address, else 507035. Tile refuses a
// sub-32-byte DMA, so the padding is squeezed out in UB, not on the way to GM.
constexpr unsigned VSTS_ALIGN = 32;
constexpr unsigned GROUP_PITCH_B16 = VSTS_ALIGN / 2u;  // in b16 elements
// RULE: vselr indices reach only the low 128 source bytes: 4 groups per gather.
constexpr unsigned GROUPS_PER_COMPACT = 4;
constexpr unsigned EVENT_SLOTS = 8;  // size of the event-id array
// the list below writes 8 ids by hand; a bigger array aliases onto EVENT_ID0
static_assert(EVENT_SLOTS == 8, "extend buffer_free's initialiser first");
constexpr unsigned UB_ALIGN = 512;
// A5 has 256 KB. Device pass only. This kernel is A5-only regardless.
constexpr unsigned UB_BYTES = PTO_UBUF_SIZE_BYTES;

// bf16 bit-field constants. bf16 is 1-8-7, so a magnitude's biased exponent is
// simply bits >> 7 once the sign is cleared.
constexpr uint16_t BF16_ABS = 0x7FFFu;  // clears the sign bit
constexpr int16_t BF16_MANT_BITS = 7;
constexpr int16_t E8M0_BIAS_ADJ = -2;  // byte = b - 2 (Algorithm 1, FLOOR)
constexpr int16_t RECIP_OFFSET = 256;  // 1/X exponent field = 256 - b
// b must stay in a window where 1/X is finite, non-subnormal bf16: field 256-b
// must land in [2, 254]. Clamp b, then derive BOTH outputs from the clamped b.
constexpr int16_t B_MIN = 2;
constexpr int16_t B_MAX = 254;

// Every derived size for one instantiation.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
struct QuantShape {
  static constexpr unsigned tile_elems = Rows * K;
  static constexpr unsigned blocks = tile_elems / MX_BLOCK;
  static constexpr unsigned in_bytes = tile_elems * 2u;
  static constexpr unsigned q_bytes = tile_elems / 2u;
  static constexpr unsigned scale_bytes = blocks;

  // One "group" is one vcgmax: 8 blocks == 256 elements, which is also what one
  // quant_pack iteration consumes, emitting 128 packed bytes.
  static constexpr unsigned groups = blocks / VCGMAX_B16_RESULTS;
  // one vselr squeezes 8 padded groups into 64 contiguous maxima
  static constexpr unsigned compact_iters = groups / GROUPS_PER_COMPACT;
  static constexpr unsigned b_iters = blocks / B16_LANES;
  static constexpr unsigned c_iters = tile_elems / (2u * B16_LANES);

  static constexpr unsigned aligned_in =
      (in_bytes + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
  static constexpr unsigned aligned_q =
      (q_bytes + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
  static constexpr unsigned aligned_s =
      (scale_bytes + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
  // padded to a 32-byte pitch, plus a register of read-ahead headroom
  static constexpr unsigned aligned_max =
      (groups * VSTS_ALIGN + B16_LANES * 2u + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
  static constexpr unsigned aligned_packed =
      (blocks * 2u + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
  static constexpr unsigned aligned_mult =
      (blocks * 2u + UB_ALIGN - 1) & ~(UB_ALIGN - 1);

  static constexpr unsigned slot_stride = aligned_in + aligned_q + aligned_s;
  static constexpr unsigned scratch_base = NBuffers * slot_stride;
  static constexpr unsigned ub_needed =
      scratch_base + aligned_max + aligned_mult;

  static_assert(K % MX_BLOCK == 0, "a block may not straddle a row boundary");
  static_assert(tile_elems % PASS_B_GRAIN == 0,
                "tile must hold a whole multiple of 128 blocks");
  static_assert(blocks % VCGMAX_B16_RESULTS == 0,
                "blocks must divide into whole vcgmax groups");
  static_assert(groups % GROUPS_PER_COMPACT == 0,
                "compaction squeezes GROUPS_PER_COMPACT groups at a time");
  static_assert(tile_elems % (2u * B16_LANES) == 0,
                "quant_pack consumes 256 elements per iteration");
  static_assert(sizeof(bfloat16_t) == 2, "RowsFor assumes 2-byte elements");
  static_assert(ub_needed <= UB_BYTES, "UB overflow");
  static_assert(NBuffers <= EVENT_SLOTS, "NBUF exceeds the event-id array");
  static_assert(NPrefetch < NBuffers,
                "PREFETCH == NBUF drains every MTE3->MTE2 token: deadlock");
};

// Byte offsets within a pipeline slot, plus shared scratch. Constexpr
// *variables*, not functions: a constexpr function cannot be called from
// [aicore] code, so the slot base is multiplied in at the use site.
template <typename Shape>
struct SlotOffset {
  static constexpr unsigned x = 0;
  static constexpr unsigned q = Shape::aligned_in;
  static constexpr unsigned s = Shape::aligned_in + Shape::aligned_q;
  static constexpr unsigned maxima = Shape::scratch_base;
  static constexpr unsigned packed = Shape::scratch_base + Shape::aligned_max;
  static constexpr unsigned mult = packed + Shape::aligned_packed;
};

#ifdef __DAV_VEC__
// A flat run of Elems values in GM, and the matching UB tile, for one dtype.
template <typename T, unsigned Elems>
using GmShape = pto::Shape<1, 1, 1, 1, Elems>;
template <typename T, unsigned Elems>
using GmStride = pto::Stride<1, 1, 1, Elems, 1>;
template <typename T, unsigned Elems>
using UbTile = Tile<TileType::Vec, T, 1, Elems, BLayout::RowMajor, 1, Elems>;

// ------------------------------------------------------- abs_block_max
// Per-32-element magnitude max. A 2:1 fold makes 16 lanes == one block,
// which is what vcgmax's group size requires. A 4:1 fold silently reports
// max(block 2j, block 2j+1) instead.
template <typename Shape>
__tf__ static AICORE void abs_block_max(__ubuf__ uint16_t *x,
                                        __ubuf__ uint16_t *maxima) {
  __VEC_SCOPE__ {
    MaskReg all = pset_b16(PAT_ALL);
    // PAT_VL8 matches VCGMAX_B16_RESULTS: a pattern predicate, like the rest of
    // this kernel, rather than the runtime CreatePredicate path.
    MaskReg out8 = pset_b16(PAT_VL8);
    vector_u16 abs_mask;
    vdup(abs_mask, BF16_ABS, all, MODE_ZEROING);

    for (uint16_t g = 0; g < (uint16_t)Shape::groups; ++g) {
      const uint32_t base = (uint32_t)g * 256u;
      vector_u16 even, odd, folded, grouped;
      vlds(even, odd, x + base, 0, DINTLV_B16);  // lane i: elements 2i, 2i+1
      vand(even, even, abs_mask, all);
      vand(odd, odd, abs_mask, all);
      // sign cleared, so a signed max over the bit patterns IS a magnitude max
      vmax((vector_s16 &)folded, (vector_s16 &)even, (vector_s16 &)odd, all);
      vcgmax((vector_s16 &)grouped, (vector_s16 &)folded,
             all);  // 8 block maxima
      // 32-byte pitch, not 16: see VSTS_ALIGN
      vsts(grouped, maxima + (uint32_t)g * GROUP_PITCH_B16, 0, NORM_B16, out8);
    }
    mem_bar(VST_VLD);
  }
}

// ------------------------------------------------------ compact_maxima
// Squeeze out the padding VSTS_ALIGN forces: output byte i takes input byte
// 2*(i & 0xF0) + (i & 0x0F). The ramp is loop-invariant.
template <typename Shape>
__tf__ static AICORE void compact_maxima(__ubuf__ uint16_t *padded,
                                         __ubuf__ uint16_t *packed) {
  __VEC_SCOPE__ {
    MaskReg all = pset_b16(PAT_ALL);
    MaskReg b8_all = pset_b8(PAT_ALL);
    MaskReg out32 = pset_b16(PAT_VL32);  // 32 b16 == 64 bytes
    vector_u8 idx, hi, lo, mask_hi, mask_lo;
    vci((vector_s8 &)idx, (int8_t)0, INC_ORDER);
    vdup(mask_hi, (uint8_t)0xF0, b8_all, MODE_ZEROING);
    vdup(mask_lo, (uint8_t)0x0F, b8_all, MODE_ZEROING);
    vand(hi, idx, mask_hi, b8_all);
    vand(lo, idx, mask_lo, b8_all);
    vadd((vector_s8 &)hi, (vector_s8 &)hi, (vector_s8 &)hi, b8_all);  // 2*hi
    vadd((vector_s8 &)idx, (vector_s8 &)hi, (vector_s8 &)lo, b8_all);

    for (uint16_t iter = 0; iter < (uint16_t)Shape::compact_iters; ++iter) {
      vector_u16 src, dst;
      const uint32_t in_off =
          (uint32_t)iter * GROUPS_PER_COMPACT * GROUP_PITCH_B16;
      vlds(src, padded + in_off, 0, NORM);
      vselr((vector_u8 &)dst, (vector_u8 &)src, idx);
      vsts(dst,
           packed + (uint32_t)iter * GROUPS_PER_COMPACT * VCGMAX_B16_RESULTS, 0,
           NORM_B16, out32);
    }
    mem_bar(VST_VLD);
  }
}

// ------------------------------------------------------ scale_and_mult
// maxima -> E8M0 scale byte + one bf16 reciprocal per block. quant_pack reads
// this array with E2B_B16, whose x16 replication matches its pair-granular
// deinterleave exactly, so no duplication is needed here.
template <typename Shape>
__tf__ static AICORE void scale_and_mult(__ubuf__ uint16_t *maxima,
                                         __ubuf__ uint16_t *mult,
                                         __ubuf__ uint16_t *scale) {
  __VEC_SCOPE__ {
    MaskReg all = pset_b16(PAT_ALL);
    vector_u16 recip_off;
    vdup(recip_off, (uint16_t)RECIP_OFFSET, all, MODE_ZEROING);

    for (uint16_t iter = 0; iter < (uint16_t)Shape::b_iters; ++iter) {
      vector_u16 amax, b, byte, recip;
      vlds(amax, maxima + (uint32_t)iter * B16_LANES, 0, NORM);
      // bit 15 is already clear, so this shift alone yields the biased exponent
      vshrs(b, amax, BF16_MANT_BITS, all, MODE_ZEROING);
      vmaxs(b, b, B_MIN, all);
      vmins(b, b, B_MAX, all);
      vadds(byte, b, E8M0_BIAS_ADJ, all);  // byte = b - 2
      vsts(byte, scale + (uint32_t)iter * 64u, 0, PK_B16, all);
      vsub(recip, recip_off, b, all);  // 1/X exponent field = 256 - b
      vshls(recip, recip, BF16_MANT_BITS, all, MODE_ZEROING);
      vsts(recip, mult + (uint32_t)iter * B16_LANES, 0, NORM_B16, all);
    }
    mem_bar(VST_VLD);
  }
}

// ---------------------------------------------------------- quant_pack
// Scale, cast, pack -- 256 elements per iteration, no gather.
//
// One vcvt turns 128 bf16 into 64 bytes deposited at byte STRIDE 4, offset
// chosen by PART_P0..P3. Converting two halves into offsets 0 and 1, OR-ing
// them and storing with PK_B32 (which keeps the low 2 bytes of each 4-byte
// group) writes 128 CONTIGUOUS bytes, so the compacting vselr disappears.
// This is CANN's CalcQuantizedFP8Values_Unroll2 shape, with one change: fp4
// puts two elements in a byte, so an element-granular DINTLV_B16 would pair
// element 4k with 4k+2. Deinterleaving at b32 -- pairs, not elements -- keeps
// (4k, 4k+1) together and restores natural order.
//
// The halves also share one multiplier register: after a pair-granular
// deinterleave both b16 lanes 2j and 2j+1 of either half belong to block j/8,
// so E2B_B16's x16 replication is exactly right and the duplicated multiplier
// array is no longer needed.
template <typename Shape>
__tf__ static AICORE void quant_pack(__ubuf__ uint16_t *x,
                                     __ubuf__ uint16_t *mult,
                                     __ubuf__ uint8_t *q) {
  __VEC_SCOPE__ {
    MaskReg all = pset_b16(PAT_ALL);
    MaskReg b8_all = pset_b8(PAT_ALL);
    MaskReg b32_all = pset_b32(PAT_ALL);

    for (uint16_t iter = 0; iter < (uint16_t)Shape::c_iters; ++iter) {
      vector_u16 mu;
      vector_u32 even, odd;
      vector_bf16 lo, hi;
      vector_f4e2m1x2 p0, p1, packed;
      vlds(mu, mult + (uint32_t)iter * VCGMAX_B16_RESULTS, 0, E2B_B16);
      vlds(even, odd, (__ubuf__ uint32_t *)x + (uint32_t)iter * B16_LANES, 0,
           DINTLV_B32);
      vmul(lo, (vector_bf16 &)even, (vector_bf16 &)mu, all);
      vmul(hi, (vector_bf16 &)odd, (vector_bf16 &)mu, all);
      vcvt(p0, lo, all, ROUND_R, PART_P0);
      vcvt(p1, hi, all, ROUND_R, PART_P1);
      vor((vector_u8 &)packed, (vector_u8 &)p0, (vector_u8 &)p1, b8_all);
      // 256 elements in, but PK_B32 keeps 2 of every 4 bytes: 128 bytes out
      vsts((vector_u16 &)packed,
           (__ubuf__ uint16_t *)(q + (uint32_t)iter * B16_LANES), 0, PK_B32,
           b32_all);
    }
    mem_bar(VST_VLD);
  }
}

// Move one tile of `T` between GM and UB. Both directions share the view setup;
// only the final TLOAD/TSTORE differs.
template <typename T, unsigned Elems, bool ToUb>
inline AICORE void transfer(uint32_t tile_index, uint32_t ub_offset,
                            __gm__ void *gm_base) {
  UbTile<T, Elems> ub;
  TASSIGN(ub, ub_offset);
  GlobalTensor<T, GmShape<T, Elems>, GmStride<T, Elems>> gm(
      (__gm__ T *)gm_base + (uint64_t)tile_index * Elems, GmShape<T, Elems>());
  if constexpr (ToUb) {
    TLOAD(ub, gm);
  } else {
    TSTORE(gm, ub);
  }
}

// Start the async load of this core's nth tile, if it has one. A function, not
// a lambda: set_flag/wait_flag do not resolve inside a lambda.
template <typename Shape, unsigned Buffers>
inline AICORE void issue_load(uint32_t nth, uint32_t core_id,
                              uint32_t core_count, uint32_t tiles,
                              const event_t *buffer_free, __gm__ void *x_gm) {
  const uint32_t tile_index = core_id + nth * core_count;
  if (tile_index >= tiles) return;
  const uint32_t buf = nth % Buffers;
  wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buf]);
  transfer<bfloat16_t, Shape::tile_elems, true>(
      tile_index, buf * Shape::slot_stride + SlotOffset<Shape>::x, x_gm);
  set_flag(PIPE_MTE2, PIPE_V, buffer_free[buf]);
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

// The pipeline: each core walks a strided subset of the tiles, keeping Prefetch
// loads in flight so DMA and the vector pipe overlap.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
__global__ AICORE void mxfp4_quant(__gm__ void *x_gm, __gm__ void *q_gm,
                                   __gm__ void *s_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using Shape = QuantShape<K, Rows, NBuffers, NPrefetch>;
  using Off = SlotOffset<Shape>;
  set_mask_norm();
  set_vector_mask(-1, -1);
  const event_t buffer_free[EVENT_SLOTS] = {EVENT_ID0, EVENT_ID1, EVENT_ID2,
                                            EVENT_ID3, EVENT_ID4, EVENT_ID5,
                                            EVENT_ID6, EVENT_ID7};
  const uint32_t core_id = get_block_idx(), core_count = get_block_num();
  const uint32_t tiles = batch / Rows;

  for (unsigned i = 0; i < NBuffers; ++i)  // every buffer starts free
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
  for (unsigned i = 0; i < NPrefetch; ++i)
    issue_load<Shape, NBuffers>(i, core_id, core_count, tiles, buffer_free,
                                x_gm);

  uint32_t issued = 0;
  for (uint32_t tile_index = core_id; tile_index < tiles;
       tile_index += core_count, ++issued) {
    const uint32_t buf = issued % NBuffers;
    // issued ahead of the wait below, so this load overlaps this tile's compute
    issue_load<Shape, NBuffers>(issued + NPrefetch, core_id, core_count, tiles,
                                buffer_free, x_gm);
    wait_flag(PIPE_MTE2, PIPE_V, buffer_free[buf]);
    const uint32_t slot = buf * Shape::slot_stride;
    abs_block_max<Shape>((__ubuf__ uint16_t *)(uintptr_t)(slot + Off::x),
                         (__ubuf__ uint16_t *)(uintptr_t)Off::maxima);
    compact_maxima<Shape>((__ubuf__ uint16_t *)(uintptr_t)Off::maxima,
                          (__ubuf__ uint16_t *)(uintptr_t)Off::packed);
    scale_and_mult<Shape>((__ubuf__ uint16_t *)(uintptr_t)Off::packed,
                          (__ubuf__ uint16_t *)(uintptr_t)Off::mult,
                          (__ubuf__ uint16_t *)(uintptr_t)(slot + Off::s));
    quant_pack<Shape>((__ubuf__ uint16_t *)(uintptr_t)(slot + Off::x),
                      (__ubuf__ uint16_t *)(uintptr_t)Off::mult,
                      (__ubuf__ uint8_t *)(uintptr_t)(slot + Off::q));
    set_flag(PIPE_V, PIPE_MTE3, buffer_free[buf]);
    wait_flag(PIPE_V, PIPE_MTE3, buffer_free[buf]);
    transfer<uint8_t, Shape::q_bytes, false>(tile_index, slot + Off::q, q_gm);
    transfer<uint8_t, Shape::scale_bytes, false>(tile_index, slot + Off::s,
                                                 s_gm);
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buf]);
  }
  for (unsigned i = 0; i < NBuffers; ++i)  // drain
    wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
#else
  (void)x_gm;
  (void)q_gm;
  (void)s_gm;
  (void)batch;
#endif
}

// ---------------------------------------------------------------- entry points
// One .so serves every K: fold over SUPPORTED_K for the instantiation.
template <std::size_t... Idx>
inline void launch_for_k(uint32_t bd, void *stream, uint8_t *x, uint8_t *q,
                         uint8_t *s, uint32_t batch, uint32_t k,
                         std::index_sequence<Idx...>) {
  ((k == SUPPORTED_K[Idx]
        ? (void)(mxfp4_quant<SUPPORTED_K[Idx], RowsFor<SUPPORTED_K[Idx]>::value,
                             DEF_BUFFERS, DEF_PREFETCH>
                 <<<bd, nullptr, stream>>>(x, q, s, batch))
        : (void)0),
   ...);
}

// An unsupported k is a silent no-op; the host validates (check_k).
extern "C" void call_mxfp4_quant(uint32_t bd, void *stream, uint8_t *x,
                                 uint8_t *q, uint8_t *s, uint32_t batch,
                                 uint32_t k) {
  launch_for_k(bd, stream, x, q, s, batch, k,
               std::make_index_sequence<SUPPORTED_COUNT>{});
}

// Default width, K = DEFAULT_K, for callers that do not choose.
extern "C" void call_mxfp4_quant_default(uint32_t bd, void *stream, uint8_t *x,
                                         uint8_t *q, uint8_t *s,
                                         uint32_t batch) {
  call_mxfp4_quant(bd, stream, x, q, s, batch, DEFAULT_K);
}

template <std::size_t... Idx>
inline uint32_t rows_for_k(uint32_t k, std::index_sequence<Idx...>) {
  uint32_t rows = 0;
  ((k == SUPPORTED_K[Idx] ? (void)(rows = RowsFor<SUPPORTED_K[Idx]>::value)
                          : (void)0),
   ...);
  return rows;  // 0 for an unsupported k; the host raises on that
}

// So the host does not have to restate the tiling rule.
extern "C" uint32_t mxfp4_rows_for(uint32_t k) {
  return rows_for_k(k, std::make_index_sequence<SUPPORTED_COUNT>{});
}
