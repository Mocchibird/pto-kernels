// Everything the two fused Hadamard + MXFP4 quantize examples have in common:
// the UB and DMA constants, the shape arithmetic, the four quantizer passes and
// the tile pipeline. What is NOT here is the butterfly -- sweep, rotate and the
// QuantShape geometry that drives them -- because that is the whole difference
// between the two kernels, one rotating a whole row and the other independent
// 32-element blocks.
//
// Included by both kernels in this directory. Each defines its own
// SUPPORTED_K, QuantShape, butterfly and entry points.
//
// The tunables (FUSED_TILE_ELEMS, FUSED_BUFFERS, FUSED_PREFETCH) and the two
// ladder switches (FUSED_ROTATE_ONLY, FUSED_NO_ROTATE) apply to both kernels
// and are documented in either README.
//
// The `static` on the moved __tf__ functions came with them from the .cpp and
// is not doing anything a template does not already do -- each example is its
// own .so, so there is one instantiation either way. It is left alone because
// removing it would change the generated code for no stated reason.
#ifndef PTO_EXAMPLES_FUSED_HADAMARD_QUANT_COMMON_HPP
#define PTO_EXAMPLES_FUSED_HADAMARD_QUANT_COMMON_HPP

#include <pto/pto-inst.hpp>
#include <type_traits>
#include <utility>
using namespace pto;

// --- butterfly geometry, from fast_hadamard_a5 -------------------------------
// WINDOW is two registers: the deinterleave load splits a 2*lanes run into
// even/odd halves, and the concat-halves store puts them back.
constexpr unsigned SLOTS = 8;  // unroll width: register sets per sweep
constexpr unsigned HAD_ALIGN = 512;

constexpr unsigned MX_BLOCK = 32;  // MXFP4 block: 32 elements, one E8M0 scale

// The three pipeline parameters, overridable for tuning. The defaults are the
// tuned point: 24576 is the largest tile that fits UB at all, and 3 is the only
// buffer count it fits at. It beats the 16384/4/2 this kernel shipped with by
// 1.063-1.066x on large launches, bit-exact, with no regression at any shape
// measured. Numbers and the full grid are in the README.
//
// Overriding is safe in the way that matters: every combination is checked by
// the static_asserts at the end of QuantShape, so a tile that will not fit UB,
// or a prefetch depth that would deadlock, fails to COMPILE rather than
// misbehaving. And the host reads rows-per-tile back from the .so
// (hadamard_mxfp4_full_rows_for), so a changed TILE_ELEMS cannot desynchronise
// from the harness.
#ifndef FUSED_BUFFERS
#define FUSED_BUFFERS 3
#endif
#ifndef FUSED_PREFETCH
#define FUSED_PREFETCH 2
#endif
#ifndef FUSED_TILE_ELEMS
#define FUSED_TILE_ELEMS 24576  // 48 KB bf16
#endif

// WHY vsts AND NOT vscatter. The butterfly halves go out as a vsts
// NORM_B16 pair. A vscatter with an identity index was built and measured
// against it -- same instruction count, same registers, same dependency
// chain, bit-identical output, the opcode the only difference -- and it cost
// +28.96 us per call, 111.57 against 82.61, paired 0.741x, resolved. That
// also refuted a ROL5 index meant to absorb the rotation fixup into the store
// for free: the fixup is 13.8% of the kernel (82.74 -> 71.31 us, paired
// 1.164x), so the opcode swap alone costs about 2.5x what it would remove.
// Break-even needed vscatter under ~5.5x a vsts. Both arms are gone; the
// numbers are why the store looks the way it does.
//
// It also explains the pure-PTO kernel: its two TTRANS calls are 96% of its
// 2052 us, and the vendor builds bf16 TTRANS out of vgather2/vscatter.

constexpr unsigned DEF_BUFFERS = FUSED_BUFFERS;    // UB pipeline buffers
constexpr unsigned DEF_PREFETCH = FUSED_PREFETCH;  // tiles in flight ahead
constexpr unsigned TILE_ELEMS = FUSED_TILE_ELEMS;
// RULE: every GM move_tile is one row and a Tile refuses a row under 32 bytes.
// The scale row is the smallest, at tile_elems/32 bytes, so a tile must be a
// whole multiple of 32*MX_BLOCK elements. DMA sets this grain, not the compute.
constexpr unsigned TILE_GRAIN = 1024;

template <unsigned A, unsigned B>
struct Gcd {
  static constexpr unsigned value = Gcd<B, A % B>::value;
};

template <unsigned A>
struct Gcd<A, 0u> {
  static constexpr unsigned value = A;
};

// ROWS_PER_TILE: the largest row count whose tile is a whole number of grains.
// Not TILE_ELEMS / K -- for a large odd factor (768 = 32*24) the quotient is
// not a multiple of the grain. Zero means inadmissible; Rows asserts on it.
//
// Rows*K is a multiple of TILE_GRAIN exactly when Rows is a multiple of
// TILE_GRAIN / gcd(K, TILE_GRAIN), so the answer is the largest such multiple
// within cap. Counting down from cap one at a time instead costs a template
// instantiation per step: at TILE_ELEMS 32768 and K=32 that is 1024 of them,
// which is the compiler's default depth limit, and the tile could not be raised
// without hitting it. This form has no such ceiling.
template <unsigned K>
struct RowsFor {
  static constexpr unsigned cap = TILE_ELEMS / K > 1u ? TILE_ELEMS / K : 1u;
  static constexpr unsigned step = TILE_GRAIN / Gcd<K, TILE_GRAIN>::value;
  static constexpr unsigned value = (cap / step) * step;
};

#ifdef __CCE_AICORE__
constexpr unsigned B16_LANES = 128;  // bf16 lanes in one vector register
// vcgmax on b16 groups 16 lanes, 8 results in lanes 0..7.
constexpr unsigned VCGMAX_B16_GROUP = 16;
constexpr unsigned VCGMAX_B16_RESULTS = B16_LANES / VCGMAX_B16_GROUP;
static_assert(VCGMAX_B16_RESULTS == 8, "block_abs_max stores with PAT_VL8");
// RULE: vsts needs a 32-byte-aligned UB address, else 507035. Tile refuses a
// sub-32-byte DMA, so the padding is squeezed out in UB, not on the way to GM.
constexpr unsigned VSTS_ALIGN = 32;
constexpr unsigned GROUP_PITCH_B16 = VSTS_ALIGN / 2u;  // in b16 elements
// RULE: vselr indices reach only the low 128 source bytes: 4 groups per gather.
constexpr unsigned GROUPS_PER_COMPACT = 4;
constexpr unsigned EVENT_SLOTS = 8;
static_assert(EVENT_SLOTS == 8, "extend buffer_free's initialiser first");
constexpr unsigned UB_ALIGN = 512;
// A5 has 256 KB.
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

// RULE: a constexpr function cannot be called from [aicore] code.
template <unsigned Value>
struct Log2 {
  static constexpr unsigned value = 1 + Log2<(Value >> 1)>::value;
};

template <>
struct Log2<1> {
  static constexpr unsigned value = 0;
};

// Largest unroll width <= Limit that divides Windows exactly, so the sweep
// covers the tile with no tail. A value template rather than a constexpr
// function: [aicore] code may not call one, even to initialise a constexpr.
template <unsigned Windows, unsigned Limit>
struct UnrollFor {
  static constexpr unsigned value =
      (Windows % Limit == 0) ? Limit : UnrollFor<Windows, Limit / 2>::value;
};

template <unsigned Windows>
struct UnrollFor<Windows, 1u> {
  static constexpr unsigned value = 1u;
};

template <unsigned Bytes>
struct RoundUp {
  static constexpr unsigned value = (Bytes + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
};

// Byte offsets within a pipeline slot, plus shared scratch. Constexpr
// *variables* for the reason above, so the slot base is multiplied in at
// the use site.
template <typename Shape>
struct SlotOffset {
  static constexpr unsigned input = 0;
  static constexpr unsigned nibbles = Shape::aligned_in;
  static constexpr unsigned scales = Shape::aligned_in + Shape::aligned_q;
  static constexpr unsigned maxima = Shape::scratch_base;
  static constexpr unsigned packed = Shape::scratch_base + Shape::aligned_max;
  static constexpr unsigned reciprocal = packed + Shape::aligned_packed;
};

#ifdef __DAV_VEC__
// A flat run of Elems values in GM, and the matching UB tile, for one dtype.
template <typename T, unsigned Elems>
using GmShape = pto::Shape<1, 1, 1, 1, Elems>;
template <typename T, unsigned Elems>
using GmStride = pto::Stride<1, 1, 1, Elems, 1>;
template <typename T, unsigned Elems>
using UbTile = Tile<TileType::Vec, T, 1, Elems, BLayout::RowMajor, 1, Elems>;
// Same tile with a RUNTIME valid column count, zero-filling the rest in UB, for
// the one partial tile a batch can end on.
template <typename T, unsigned Elems>
using UbTilePart =
    Tile<TileType::Vec, T, 1, Elems, BLayout::RowMajor, 1, DYNAMIC,
         SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Zero>;

using HadRegs = vector_u16[SLOTS];

// ------------------------------------------------------- block_abs_max
// Per-32-element magnitude max. A 2:1 fold makes 16 lanes == one block,
// which is what vcgmax's group size requires. A 4:1 fold silently reports
// max(block 2j, block 2j+1) instead.

template <typename Shape>
__tf__ static AICORE void block_abs_max(__ubuf__ uint16_t *input,
                                        __ubuf__ uint16_t *maxima) {
  __VEC_SCOPE__ {
    MaskReg all_lanes = pset_b16(PAT_ALL);
    // PAT_VL8 matches VCGMAX_B16_RESULTS
    MaskReg low_eight = pset_b16(PAT_VL8);
    vector_u16 abs_mask;
    vdup(abs_mask, BF16_ABS, all_lanes, MODE_ZEROING);

    for (uint16_t group = 0; group < (uint16_t)Shape::groups; ++group) {
      const uint32_t base = (uint32_t)group * 256u;
      vector_u16 even, odd, folded, grouped;
      vlds(even, odd, input + base, 0,
           DINTLV_B16);  // lane i: elements 2i, 2i+1
      vand(even, even, abs_mask, all_lanes);
      vand(odd, odd, abs_mask, all_lanes);
      // sign cleared, so a signed max over the bit patterns IS a magnitude max
      vmax((vector_s16 &)folded, (vector_s16 &)even, (vector_s16 &)odd,
           all_lanes);
      vcgmax((vector_s16 &)grouped, (vector_s16 &)folded, all_lanes);
      // 32-byte pitch, not 16: see VSTS_ALIGN
      vsts(grouped, maxima + (uint32_t)group * GROUP_PITCH_B16, 0, NORM_B16,
           low_eight);
    }
    mem_bar(VST_VLD);
  }
}

// ------------------------------------------------------ compact_maxima
// Squeeze out the padding VSTS_ALIGN forces: output byte i takes input byte
// 2*(i & 0xF0) + (i & 0x0F).
template <typename Shape>
__tf__ static AICORE void compact_maxima(__ubuf__ uint16_t *padded,
                                         __ubuf__ uint16_t *packed) {
  __VEC_SCOPE__ {
    MaskReg all_byte_lanes = pset_b8(PAT_ALL);
    MaskReg low_32 = pset_b16(PAT_VL32);  // 32 b16 == 64 bytes
    vector_u8 byte_index, high_half, low_half, high_mask, low_mask;
    vci((vector_s8 &)byte_index, (int8_t)0, INC_ORDER);
    vdup(high_mask, (uint8_t)0xF0, all_byte_lanes, MODE_ZEROING);
    vdup(low_mask, (uint8_t)0x0F, all_byte_lanes, MODE_ZEROING);
    vand(high_half, byte_index, high_mask, all_byte_lanes);
    vand(low_half, byte_index, low_mask, all_byte_lanes);
    vadd((vector_s8 &)high_half, (vector_s8 &)high_half, (vector_s8 &)high_half,
         all_byte_lanes);  // 2*high_half
    vadd((vector_s8 &)byte_index, (vector_s8 &)high_half, (vector_s8 &)low_half,
         all_byte_lanes);

    for (uint16_t gather = 0; gather < (uint16_t)Shape::compact_iters;
         ++gather) {
      vector_u16 padded_chunk, packed_chunk;
      const uint32_t src_offset =
          (uint32_t)gather * GROUPS_PER_COMPACT * GROUP_PITCH_B16;
      vlds(padded_chunk, padded + src_offset, 0, NORM);
      vselr((vector_u8 &)packed_chunk, (vector_u8 &)padded_chunk, byte_index);
      vsts(packed_chunk,
           packed + (uint32_t)gather * GROUPS_PER_COMPACT * VCGMAX_B16_RESULTS,
           0, NORM_B16, low_32);
    }
    mem_bar(VST_VLD);
  }
}

// -------------------------------------------------------- derive_scales
// maxima -> E8M0 scale byte + one bf16 reciprocal per block. pack_nibbles
// reads this array with E2B_B16, whose x16 replication matches its
// pair-granular deinterleave exactly, so no duplication is needed here.
template <typename Shape>
__tf__ static AICORE void derive_scales(__ubuf__ uint16_t *maxima,
                                        __ubuf__ uint16_t *recips_out,
                                        __ubuf__ uint16_t *scale_out) {
  __VEC_SCOPE__ {
    MaskReg all_lanes = pset_b16(PAT_ALL);
    vector_u16 bias;
    vdup(bias, (uint16_t)RECIP_OFFSET, all_lanes, MODE_ZEROING);

    for (uint16_t chunk = 0; chunk < (uint16_t)Shape::b_iters; ++chunk) {
      vector_u16 block_max, exponent, scale_byte, reciprocal;
      vlds(block_max, maxima + (uint32_t)chunk * B16_LANES, 0, NORM);
      // bit 15 is already clear, so this shift alone yields the biased exponent
      vshrs(exponent, block_max, BF16_MANT_BITS, all_lanes, MODE_ZEROING);
      vmaxs(exponent, exponent, B_MIN, all_lanes);
      vmins(exponent, exponent, B_MAX, all_lanes);
      vadds(scale_byte, exponent, E8M0_BIAS_ADJ, all_lanes);
      vsts(scale_byte, scale_out + (uint32_t)chunk * 64u, 0, PK_B16, all_lanes);
      vsub(reciprocal, bias, exponent, all_lanes);
      vshls(reciprocal, reciprocal, BF16_MANT_BITS, all_lanes, MODE_ZEROING);
      vsts(reciprocal, recips_out + (uint32_t)chunk * B16_LANES, 0, NORM_B16,
           all_lanes);
    }
    mem_bar(VST_VLD);
  }
}

// --------------------------------------------------------- pack_nibbles
// Scale, cast, pack -- 256 elements per iteration, no gather.
// One vcvt puts 64 bytes at byte STRIDE 4, offset chosen by
// PART_P0..P3, so converting two halves into offsets 0 and 1, OR-ing, and
// storing with PK_B32 (keeps the low 2 bytes of each 4-byte group) writes 128
// CONTIGUOUS bytes. RULE: fp4 packs two elements per byte, so DINTLV_B16 would
// pair element 4k with 4k+2 -- deinterleave at b32 (pairs) to keep (4k, 4k+1)
// together. That also puts both b16 lanes of a half in block j/8, so E2B_B16's
// x16 replication is exact and one multiplier register serves both halves.
template <typename Shape>
__tf__ static AICORE void pack_nibbles(__ubuf__ uint16_t *input,
                                       __ubuf__ uint16_t *reciprocal,
                                       __ubuf__ uint8_t *nibble_out) {
  __VEC_SCOPE__ {
    MaskReg all_lanes = pset_b16(PAT_ALL);
    MaskReg all_byte_lanes = pset_b8(PAT_ALL);
    MaskReg all_b32_lanes = pset_b32(PAT_ALL);

    for (uint16_t chunk = 0; chunk < (uint16_t)Shape::c_iters; ++chunk) {
      vector_u16 recips;
      vector_u32 even, odd;
      vector_bf16 scaled_even, scaled_odd;
      vector_f4e2m1x2 packed_even, packed_odd, packed;
      vlds(recips, reciprocal + (uint32_t)chunk * VCGMAX_B16_RESULTS, 0,
           E2B_B16);
      vlds(even, odd, (__ubuf__ uint32_t *)input + (uint32_t)chunk * B16_LANES,
           0, DINTLV_B32);
      vmul(scaled_even, (vector_bf16 &)even, (vector_bf16 &)recips, all_lanes);
      vmul(scaled_odd, (vector_bf16 &)odd, (vector_bf16 &)recips, all_lanes);
      vcvt(packed_even, scaled_even, all_lanes, ROUND_R, PART_P0);
      vcvt(packed_odd, scaled_odd, all_lanes, ROUND_R, PART_P1);
      vor((vector_u8 &)packed, (vector_u8 &)packed_even,
          (vector_u8 &)packed_odd, all_byte_lanes);
      // 256 elements in, but PK_B32 keeps 2 of every 4 bytes: 128 bytes out
      vsts((vector_u16 &)packed,
           (__ubuf__ uint16_t *)(nibble_out + (uint32_t)chunk * B16_LANES), 0,
           PK_B32, all_b32_lanes);
    }
    mem_bar(VST_VLD);
  }
}

// Move one tile of `T` between GM and UB. Partial carries only `valid`
// elements: the load zero-fills the rest of the UB tile so the compute passes
// still see whole registers, and the store truncates so padding never reaches
// GM.
template <typename T, unsigned Elems, bool ToUb, bool Partial = false>
inline AICORE void move_tile(uint32_t tile_index, uint32_t ub_offset,
                             __gm__ void *gm_base, uint32_t valid = 0) {
  std::conditional_t<Partial, UbTilePart<T, Elems>, UbTile<T, Elems>> ub;
  TASSIGN(ub, ub_offset);
  if constexpr (Partial) ub.ColMaskInternal = (int)valid;
  GlobalTensor<T, GmShape<T, Elems>, GmStride<T, Elems>> gm(
      (__gm__ T *)gm_base + (uint64_t)tile_index * Elems, GmShape<T, Elems>());
  if constexpr (ToUb) {
    TLOAD(ub, gm);
  } else {
    TSTORE(gm, ub);
  }
}

// Start the async load of this core's nth tile, if it has one. A function,
// not a lambda: set_flag/wait_flag do not resolve inside a lambda.
template <typename Shape, unsigned Buffers>
inline AICORE void issue_tile_load(uint32_t nth_tile, uint32_t core_id,
                                   uint32_t core_count, uint32_t tiles,
                                   uint32_t full_tiles, uint32_t tail_elems,
                                   const event_t *buffer_free,
                                   __gm__ void *input_gm) {
  const uint32_t tile_index = core_id + nth_tile * core_count;
  if (tile_index >= tiles) return;
  const uint32_t buffer = nth_tile % Buffers;
  const uint32_t off = buffer * Shape::slot_stride + SlotOffset<Shape>::input;
  wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buffer]);
  // at most one tile is partial, and only when batch does not fill it
  if (tile_index == full_tiles) {
    move_tile<bfloat16_t, Shape::tile_elems, true, true>(tile_index, off,
                                                         input_gm, tail_elems);
  } else {
    move_tile<bfloat16_t, Shape::tile_elems, true>(tile_index, off, input_gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, buffer_free[buffer]);
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

#endif  // PTO_EXAMPLES_FUSED_HADAMARD_QUANT_COMMON_HPP
