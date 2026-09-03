// Block-32 Hadamard fused with MXFP4 quantization, one launch.
//
//   x  ->  (x @ H) -> E2M1 nibbles + one E8M0 scale per 32
//
// Unfused this is two passes over HBM: read x / write rotated, then read
// rotated / write nibbles+scales. Fused it is read x / write nibbles+scales, so
// on a DMA-bound op the saving is close to the whole second pass.
//
// Built from two kernels that are already measured and merged upstream:
// fast_hadamard_a5 supplies the butterfly, mxfp4_quant_a5 the four quant
// passes, the tiling and the outputs. Both are left doing what they already do;
// what is new here is that the rotated tile never leaves UB.
//
// The butterfly was fp16 upstream and is bf16 here, which costs nothing
// structurally: vlds/vsts are bit-width ops on vector_u16 (DINTLV_B16 /
// NORM_B16), so only the arithmetic type changes, by reference cast. That is
// the same idiom mxfp4_quant_a5 already uses for its max reduction.
//
// The difference from v1 is not just a pinned width. v1 rotates a whole row, so
// K must be a power of two and at most 2048. Here the Hadamard is always 32
// wide and a row is a sequence of independent 32-blocks, which decouples the
// rotation from the row width: K goes back to any multiple of 32, so 4096 and
// 11008 work, which a row-wide rotation rejects.
//
// It falls out of the tile being a flat run of Rows*K elements. Blocks are
// contiguous and 32 long, the butterfly window is 256 = eight blocks, and
// blocks are independent -- so one window covers eight of them and never has to
// care whether they came from the same row.
//
// Also: the MXFP4 group is 32 and the Hadamard block is 32, so a scale covers
// exactly one rotated block. No reshaping, and no group straddling a rotation.
#include <pto/pto-inst.hpp>
#include <type_traits>
#include <utility>
using namespace pto;

// Row widths with an instantiation. The full set the quantizer supports: a
// 32-wide rotation puts no power-of-two constraint on the row, so 4096 and
// 11008-style widths are back. Also add any new width to the jit helper.
constexpr unsigned SUPPORTED_K[] = {32,   64,   96,   128,  192,  256,   512,
                                    768,  896,  1024, 1152, 1280, 1408,  1536,
                                    1664, 1792, 2048, 2560, 2816, 3072,  3584,
                                    4096, 5120, 6144, 7168, 8192, 14336, 16384};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_K) / sizeof(SUPPORTED_K[0]);
// --- butterfly geometry, from fast_hadamard_a5 -------------------------------
// WINDOW is two registers: the deinterleave load splits a 2*lanes run into
// even/odd halves, and the concat-halves store puts them back.
constexpr unsigned SLOTS = 8;  // unroll width: register sets per sweep
constexpr unsigned HAD_ALIGN = 512;
constexpr unsigned HAD_BLOCK = 32;  // the Hadamard block, == MX_BLOCK

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
// (hadamard_mxfp4_b32_rows_for), so a changed TILE_ELEMS cannot desynchronise
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

// Store the butterfly halves with vscatter instead of vsts.
//
//   0  vsts NORM_B16 pair -- what ships, and the default.
//   1  vscatter with an IDENTITY index. Same instruction count, same registers,
//      same dependency chain, bit-identical output; the ONLY difference is the
//      opcode. This exists to price vscatter against vsts, because vscatter's
//      cost on A5 is not documented and the vendor's bf16 TTRANS is built from
//      it -- and two TTRANS calls are 96% of the pure-PTO kernel's cost, so it
//      could be far dearer than a plain store.
//
// Mode 2 was a ROL5 index meant to absorb the rotation fixup into the store at
// no extra instruction -- the fixup is 13.8% of the kernel (82.74 -> 71.31 us
// with -DFUSED_NO_ROTFIX, paired 1.164x, resolved). It is GONE, refuted by
// mode 1: vscatter costs +28.96 us per call against vsts (111.57 against 82.61,
// paired 0.741x, resolved), so the opcode swap alone costs 2.5x what the fixup
// it would remove is worth. Break-even needed vscatter under ~5.5x a vsts.
//
// This also explains the pure-PTO kernel: its two TTRANS calls are 96% of its
// 2052 us, and the vendor builds bf16 TTRANS out of vgather2/vscatter.

#ifndef FUSED_SCATTER
#define FUSED_SCATTER 0
#endif
#if FUSED_SCATTER > 1
#error "FUSED_SCATTER=2 (ROL5 store) was measured and refuted; see above"
#endif

constexpr unsigned DEF_BUFFERS = FUSED_BUFFERS;    // UB pipeline buffers
constexpr unsigned DEF_PREFETCH = FUSED_PREFETCH;  // tiles in flight ahead
constexpr unsigned TILE_ELEMS = FUSED_TILE_ELEMS;
// RULE: every GM move_tile is one row and a Tile refuses a row under 32 bytes.
// The scale row is the smallest, at tile_elems/32 bytes, so a tile must be a
// whole multiple of 32*MX_BLOCK elements. DMA sets this grain, not the compute.
constexpr unsigned TILE_GRAIN = 1024;
// ROWS_PER_TILE: the largest row count whose tile is a whole number of grains.
// Not TILE_ELEMS / K -- for a large odd factor (768 = 32*24) the quotient is
// not a multiple of the grain. Zero means inadmissible; Rows asserts on it.
template <unsigned A, unsigned B>
struct Gcd {
  static constexpr unsigned value = Gcd<B, A % B>::value;
};
template <unsigned A>
struct Gcd<A, 0u> {
  static constexpr unsigned value = A;
};

// Rows*K is a multiple of TILE_GRAIN exactly when Rows is a multiple of
// TILE_GRAIN / gcd(K, TILE_GRAIN), so the answer is the largest such multiple
// within cap. Counting down from cap one step at a time costs a template
// instantiation per step, which is 768 of them at K=32 and would grow past the
// compiler's depth limit if the tile were raised.
template <unsigned K>
struct RowsFor {
  static constexpr unsigned cap = TILE_ELEMS / K > 1u ? TILE_ELEMS / K : 1u;
  static constexpr unsigned step = TILE_GRAIN / Gcd<K, TILE_GRAIN>::value;
  static constexpr unsigned value = (cap / step) * step;
};

#if defined(FUSED_ROTATE_ONLY) && defined(MXFP4_TQUANT)
#error "FUSED_ROTATE_ONLY does nothing in a TQuant build: TQuant owns them"
#endif

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

// Every derived size for one instantiation.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
struct QuantShape {
  static constexpr unsigned tile_elems = Rows * K;
  static constexpr unsigned blocks = tile_elems / MX_BLOCK;
  static constexpr unsigned in_bytes = tile_elems * 2u;
  static constexpr unsigned q_bytes = tile_elems / 2u;
  static constexpr unsigned scale_bytes = blocks;

  // One "group" is one vcgmax: 8 blocks == 256 elements.
  static constexpr unsigned groups = blocks / VCGMAX_B16_RESULTS;
  // Round UP: the last bite may be partial. Safe only because the buffers
  // below are sized from these counts, and nothing reads what a partial bite
  // writes past `blocks`. The asserts at the end of this struct pin both.
  static constexpr unsigned compact_iters =
      (groups + GROUPS_PER_COMPACT - 1u) / GROUPS_PER_COMPACT;
  static constexpr unsigned b_iters = (blocks + B16_LANES - 1u) / B16_LANES;
  static constexpr unsigned c_iters = tile_elems / (2u * B16_LANES);

  // maxima_bytes carries a register of read-ahead: the gather reads one
  // register past its last input offset. The rest are sized from what the loops
  // WRITE, since the two rounded counts can exceed their data by one bite.
  static constexpr unsigned maxima_bytes =
      compact_iters * GROUPS_PER_COMPACT * GROUP_PITCH_B16 * 2u + B16_LANES;
  static constexpr unsigned packed_bytes =
      compact_iters * GROUPS_PER_COMPACT * VCGMAX_B16_RESULTS * 2u;
  static constexpr unsigned aligned_in = RoundUp<in_bytes>::value;
  static constexpr unsigned aligned_q = RoundUp<q_bytes>::value;
  static constexpr unsigned aligned_s = RoundUp<b_iters * B16_LANES>::value;
  static constexpr unsigned aligned_max = RoundUp<maxima_bytes>::value;
  static constexpr unsigned aligned_packed = RoundUp<packed_bytes>::value;
  static constexpr unsigned aligned_mult =
      RoundUp<b_iters * B16_LANES * 2u>::value;

  static constexpr unsigned slot_stride = aligned_in + aligned_q + aligned_s;
  static constexpr unsigned scratch_base = NBuffers * slot_stride;
  // every scratch region SlotOffset hands out, in the same order, so the two
  // cannot drift: omitting one here silently shrinks the UB-overflow guard
  static constexpr unsigned ub_needed =
      scratch_base + aligned_max + aligned_packed + aligned_mult;

#ifdef MXFP4_TQUANT
  // TQuant reads its per-block maxima a whole register at a time, so the span
  // rounds up to b_iters registers, and its reducer flushes one 32-byte block
  // past the last group.
  static constexpr unsigned tquant_max_elems = b_iters * B16_LANES;
  static constexpr unsigned tquant_max_bytes =
      tquant_max_elems * 2u + VSTS_ALIGN;
  static constexpr unsigned tquant_scaling_bytes = blocks * 2u;
  static_assert(tquant_max_bytes <= aligned_max,
                "TQuant maxima do not fit the maxima region");
  static_assert(tquant_scaling_bytes <= aligned_mult,
                "TQuant scaling does not fit the reciprocal region");
  // numGroups truncates inside TQuant, and a partial 8-group window takes a
  // different store path.
  static_assert(tile_elems % MX_BLOCK == 0, "TQuant would drop a group");
  static_assert(blocks % 8u == 0, "TQuant would take its vstus tail path");
#endif

  // --- butterfly geometry over 32-element blocks
  // ------------------------------ Fixed at HAD_BLOCK, independent of K: the
  // tile is a flat run of Rows*K elements, so it is (Rows*K)/32 independent
  // blocks and the window packs eight of them.
  static constexpr unsigned had_lanes_b16 = B16_LANES;
  static constexpr unsigned had_window = 2u * had_lanes_b16;
  static constexpr unsigned log2_block = Log2<HAD_BLOCK>::value;
  static constexpr unsigned log2_window = Log2<had_window>::value;
  static constexpr unsigned rows_per_window = had_window / HAD_BLOCK;
  static constexpr unsigned had_group = HAD_BLOCK * rows_per_window;
  static constexpr unsigned rotations = log2_window - log2_block;
  static constexpr unsigned upper = had_group / 2u;
  static constexpr unsigned lanes =
      upper < had_lanes_b16 ? upper : had_lanes_b16;
  static constexpr unsigned chunks = upper / lanes;
  static constexpr unsigned windows_per_tile = tile_elems / had_group;
  static constexpr unsigned groups_per_iter =
      UnrollFor<windows_per_tile, SLOTS>::value;
  static constexpr unsigned had_blocks_per_tile = tile_elems / HAD_BLOCK;
  static constexpr unsigned had_iters =
      tile_elems / had_group / groups_per_iter;
  static constexpr unsigned sweep_stride = groups_per_iter * had_group;
  // How many register slots one sweep call must use. This has to be derived
  // from groups_per_iter, NOT from SLOTS.
  //
  // A slot addresses window `Slot / chunks` at chunk `Slot % chunks`, so a call
  // with N slots covers N/chunks windows, while the loop advances
  // groups_per_iter windows per iteration. Instantiating the sweep with SLOTS
  // when groups_per_iter is smaller makes consecutive iterations overlap and
  // runs the last one off the end of the tile.
  //
  // The row-wide variant cannot hit this because it defines
  // groups_per_iter = SLOTS / chunks, which makes the two agree by
  // construction. This kernel picks groups_per_iter with UnrollFor instead --
  // deliberately, because a 32-wide rotation's window count need not be a
  // multiple of 8 -- and that is exactly what decoupled them.
  static constexpr unsigned sweep_slots = groups_per_iter * chunks;

  static_assert(HAD_BLOCK == MX_BLOCK,
                "a scale must cover exactly one rotated block");
  static_assert(had_window % HAD_BLOCK == 0,
                "the butterfly window must be whole blocks");
  static_assert(tile_elems % had_group == 0,
                "tile must be a whole number of butterfly windows");
  static_assert(windows_per_tile % groups_per_iter == 0,
                "UnrollFor must divide the window count exactly");
  static_assert(chunks == 1u, "a 32-wide rotation packs; it never chunks");
  static_assert(sweep_slots <= SLOTS,
                "a sweep would need more register slots than SLOTS declares");
  // The tiling condition the overlap bug violated: one iteration's slots must
  // cover exactly the windows the stride advances, no more and no fewer.
  static_assert(sweep_slots / chunks * had_group == sweep_stride,
                "sweep slots and sweep_stride disagree: iterations overlap");

  static_assert(Rows > 0, "no Rows makes Rows*K a whole TILE_GRAIN: bad K");
  static_assert(K % MX_BLOCK == 0, "a block may not straddle a row boundary");
  static_assert(TILE_GRAIN == VSTS_ALIGN * MX_BLOCK, "grain != scale DMA row");
  static_assert(tile_elems % TILE_GRAIN == 0, "tile is not a whole grain");
  static_assert(tile_elems % (2u * B16_LANES) == 0, "pack_nibbles wants 256");
  static_assert(scale_bytes % VSTS_ALIGN == 0, "scale row is not a legal DMA");
  static_assert(q_bytes % VSTS_ALIGN == 0, "nibble row is not a legal DMA");
  static_assert(in_bytes % VSTS_ALIGN == 0, "input row is not a legal DMA");
  static_assert(blocks % VCGMAX_B16_RESULTS == 0,
                "blocks != whole vcgmax groups");
  // the rounded-up passes run one bite past the data; prove they stay inside
  static_assert(b_iters * B16_LANES <= aligned_s, "scale tail overruns");
  static_assert(b_iters * B16_LANES * 2u <= aligned_mult,
                "recips tail overruns");
  static_assert(packed_bytes <= aligned_packed, "compaction tail overruns");
  static_assert(groups * VSTS_ALIGN <= aligned_max, "padded maxima overrun");
  static_assert(
      c_iters * VCGMAX_B16_RESULTS <= b_iters * B16_LANES,
      "pack_nibbles would index recips past what derive_scales wrote");
  static_assert(sizeof(bfloat16_t) == 2, "RowsFor assumes 2-byte elements");
  static_assert(ub_needed <= UB_BYTES, "UB overflow");
  // Strictly less, not <=. The once-per-launch D load signals on EVENT_ID7 over
  // MTE2 -> V, and buffer_free[7] is also EVENT_ID7 on that same pipe pair, so
  // at NBuffers == EVENT_SLOTS buffer 7 and the D preamble would share a
  // channel. Unreachable at the shipped NBuffers of 3; this stops it becoming
  // reachable.
  static_assert(NBuffers < EVENT_SLOTS,
                "NBUF must leave EVENT_ID7 free for the D preamble");
  static_assert(NPrefetch < NBuffers,
                "PREFETCH == NBUF deadlocks the pipeline");
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

// ------------------------------------------------------- block_abs_max
// Per-32-element magnitude max. A 2:1 fold makes 16 lanes == one block,
// which is what vcgmax's group size requires. A 4:1 fold silently reports
// max(block 2j, block 2j+1) instead.
// --- the rotation ------------------------------------------------------------
// One sweep: deinterleave-load a window, add/sub, and store the halves back
// concatenated. Registers are vector_u16 because vlds/vsts are bit-width ops;
// the arithmetic type is chosen by reference cast, which is how bf16 costs
// nothing here. All loads precede all stores, which the comma fold guarantees
// by evaluating left to right -- required, not stylistic, since a store would
// otherwise clobber a window a later load still needs.
using HadRegs = vector_u16[SLOTS];

template <typename Shape, unsigned Rotations, std::size_t... Slot>
inline AICORE void sweep(__ubuf__ uint16_t *tile, uint32_t base, MaskReg all,
                         HadRegs &even, HadRegs &odd, HadRegs &sum,
                         HadRegs &diff, vector_u16 &idx_lo, vector_u16 &idx_hi,
                         std::index_sequence<Slot...>) {
  constexpr unsigned g = Shape::had_group, up = Shape::upper;
  constexpr unsigned ln = Shape::lanes, ch = Shape::chunks;
  (vlds(even[Slot], odd[Slot],
        tile + base + Slot / ch * g + Slot % ch * 2u * ln, 0, DINTLV_B16),
   ...);
  (vadd((vector_bf16 &)sum[Slot], (vector_bf16 &)even[Slot],
        (vector_bf16 &)odd[Slot], all),
   ...);
  (vsub((vector_bf16 &)diff[Slot], (vector_bf16 &)even[Slot],
        (vector_bf16 &)odd[Slot], all),
   ...);
  // A 256-element window packs eight independent 32-blocks, which leaves the
  // result rotated right by log2(window/block) = 3; these register-only
  // deinterleaves undo it, fused into the final stage using the pair that is
  // dead by then.
  //
  // ATTRIBUTION SWITCH. vdintlv was measured at ~20x a vadd, and there are
  // Rotations per slot here against five arithmetic ops, so this fixup may be
  // most of the butterfly's cost. -DFUSED_NO_ROTFIX drops the deinterleaves and
  // keeps everything else, including both stores, so the difference is the
  // fixup. It PRODUCES WRONG OUTPUT -- the registers selected below then hold
  // loaded values rather than rotated ones -- so it is for timing only and the
  // benchmark's correctness gate will reject it.
#ifndef FUSED_NO_ROTFIX
  if constexpr (Rotations >= 1) {
    (vdintlv(even[Slot], odd[Slot], sum[Slot], diff[Slot]), ...);
  }
  if constexpr (Rotations >= 2) {
    (vdintlv(sum[Slot], diff[Slot], even[Slot], odd[Slot]), ...);
  }
  if constexpr (Rotations >= 3) {
    (vdintlv(even[Slot], odd[Slot], sum[Slot], diff[Slot]), ...);
  }
#endif
  HadRegs &lo = (Rotations % 2 == 1) ? even : sum;
  HadRegs &hi = (Rotations % 2 == 1) ? odd : diff;
#if FUSED_SCATTER == 0
  (vsts(lo[Slot], tile + base + Slot / ch * g + Slot % ch * ln, 0, NORM_B16,
        all),
   ...);
  (vsts(hi[Slot], tile + base + Slot / ch * g + up + Slot % ch * ln, 0,
        NORM_B16, all),
   ...);
#else
  // vscatter instead of vsts, one for one. The index decides which experiment
  // this is; see the FUSED_SCATTER comment at the top of the file. Both halves
  // address the SAME window base, because with a permuting index the upper half
  // is no longer a contiguous run at +upper.
  (vscatter(lo[Slot], tile + base + Slot / ch * g, idx_lo, all), ...);
  (vscatter(hi[Slot], tile + base + Slot / ch * g, idx_hi, all), ...);
  (void)up;
#endif
}

// log2(K) stages over the tile already in UB, in place. The quant passes read
// the same buffer straight afterwards, which is the point of the fusion.
template <typename Shape>
__tf__ static AICORE void rotate(__ubuf__ uint16_t *tile) {
  // sweep_slots, not SLOTS: see the derivation in Shape. The register arrays
  // are sized SLOTS and a shorter pack leaves the top ones unused.
  constexpr auto slots = std::make_index_sequence<Shape::sweep_slots>{};
  constexpr unsigned plain = Shape::log2_block - (Shape::rotations ? 1u : 0u);
  __VEC_SCOPE__ {
    uint32_t lane_count = Shape::lanes;
    MaskReg all = CreatePredicate<bfloat16_t>(lane_count);
    vector_u16 even[SLOTS], odd[SLOTS], sum[SLOTS], diff[SLOTS];
    // Scatter indices, built once per tile and unused when FUSED_SCATTER == 0.
    // There is no vands, so the low three bits of the lane come out as
    // l - (l >> 3) * 8. vshrs/vmuls/vadds are vector-scalar; vadd/vsub are
    // vector-vector.
    vector_u16 idx_lo, idx_hi;
    vci((vector_s16 &)idx_lo, (int16_t)0, INC_ORDER);
#if FUSED_SCATTER == 1
    // IDENTITY index: byte-for-byte what the vsts pair does, so the output must
    // be bit-identical and the only difference measured is the opcode price.
    vdup(idx_hi, (uint16_t)Shape::upper, all, MODE_ZEROING);
    vadd(idx_hi, idx_lo, idx_hi, all);
#endif
    // Step by a literal 1 with the stride folded into base: the loop analyser
    // only verifies a tripcount for a literal step, and 1 divides any bound, so
    // had_iters may be template-dependent.
    for (uint16_t stage = 0; stage < (uint16_t)plain; ++stage) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::had_iters; ++iter)
        sweep<Shape, 0>(tile, (uint32_t)iter * Shape::sweep_stride, all, even,
                        odd, sum, diff, idx_lo, idx_hi, slots);
      mem_bar(VST_VLD);
    }
    if constexpr (Shape::rotations > 0) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::had_iters; ++iter)
        sweep<Shape, Shape::rotations>(
            tile, (uint32_t)iter * Shape::sweep_stride, all, even, odd, sum,
            diff, idx_lo, idx_hi, slots);
      mem_bar(VST_VLD);
    }
  }
}

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

#ifdef MXFP4_TQUANT
// Requires PTO 9.1.0: 9.0.0 has no MXFP4 quantizer. Included here, not at file
// scope, because this region is inside the device-pass guard.
#include <pto/npu/a5/TQuant.hpp>

// ------------------------------------------------------- tquant_passes
// One vendor tile op in place of block_abs_max, compact_maxima, derive_scales
// and pack_nibbles. validCols is tile_elems even on the partial tile: the load
// already zero-fills the pad, and a short validCols would send TQuant's own
// ZeroPadSourceTile over the input slot. Offsets::packed is left allocated and
// unused, since reclaiming it would move slot_stride.
template <typename Shape>
inline AICORE void tquant_passes(uint32_t input_offset, uint32_t nibble_offset,
                                 uint32_t scale_offset) {
  static_assert(sizeof(float4_e2m1x2_t) == 1,
                "the nibble tile assumes one byte per float4_e2m1x2_t");
  static_assert(REPEAT_BYTE / sizeof(bfloat16_t) == B16_LANES,
                "tquant_max_elems assumes a 128-lane b16 vector");
  UbTile<bfloat16_t, Shape::tile_elems> source;
  UbTile<float4_e2m1x2_t, Shape::q_bytes> nibbles;
  UbTile<uint8_t, Shape::scale_bytes> scales;
  UbTile<bfloat16_t, Shape::tquant_max_elems> block_max;
  UbTile<bfloat16_t, Shape::blocks> reciprocal;
  TASSIGN(source, input_offset);
  TASSIGN(nibbles, nibble_offset);
  TASSIGN(scales, scale_offset);
  TASSIGN(block_max, SlotOffset<Shape>::maxima);
  TASSIGN(reciprocal, SlotOffset<Shape>::reciprocal);
  // TEMPLATE order is Out, Src, Exp, Max, Scaling; ARGUMENT order is dst, exp,
  // max, scaling, src. PTO 9.1.0 release inserted a `bool Exp2DStrided` second
  // template parameter that 9.1.0-beta.3 does not have; the tile types are in a
  // non-deduced position, so neither spelling can be dropped. benchmark.py
  // compiles both and keeps whichever the local headers accept.
#ifdef MXFP4_TQUANT_EXP2D
  TQuant_MXFP4_E2M1_Impl<QuantScaleAlg::OCP, false, decltype(nibbles),
                         decltype(source), decltype(scales),
                         decltype(block_max), decltype(reciprocal)>(
      nibbles.data(), scales.data(), block_max.data(), reciprocal.data(),
      source.data(), 1u, Shape::tile_elems);
#else
  TQuant_MXFP4_E2M1_Impl<QuantScaleAlg::OCP, decltype(nibbles),
                         decltype(source), decltype(scales),
                         decltype(block_max), decltype(reciprocal)>(
      nibbles.data(), scales.data(), block_max.data(), reciprocal.data(),
      source.data(), 1u, Shape::tile_elems);
#endif
}
#endif  // MXFP4_TQUANT

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

// The pipeline: each core walks a strided subset of the tiles, keeping Prefetch
// loads in flight so DMA and the vector pipe overlap.
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
// A device function rather than the kernel body, so a caller that wants the
// pipeline over a sub-range can reach it directly. mxfp4_quant below is the
// entry point and the only caller here.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
inline AICORE void quant_tiles(__gm__ void *input_gm, __gm__ void *nibble_gm,
                               __gm__ void *scale_gm, uint32_t batch) {
  using Shape = QuantShape<K, Rows, NBuffers, NPrefetch>;
  using Offsets = SlotOffset<Shape>;
  set_mask_norm();
  set_vector_mask(-1, -1);
  const event_t buffer_free[EVENT_SLOTS] = {EVENT_ID0, EVENT_ID1, EVENT_ID2,
                                            EVENT_ID3, EVENT_ID4, EVENT_ID5,
                                            EVENT_ID6, EVENT_ID7};
  const uint32_t core_id = get_block_idx(), core_count = get_block_num();
  // the remainder, if any, rides along as one extra partial tile
  const uint32_t full_tiles = batch / Rows;
  const uint32_t tail_elems = (batch % Rows) * K;
  const uint32_t tiles = full_tiles + (tail_elems ? 1u : 0u);

  for (unsigned i = 0; i < NBuffers; ++i)  // every buffer starts free
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
  for (unsigned i = 0; i < NPrefetch; ++i)
    issue_tile_load<Shape, NBuffers>(i, core_id, core_count, tiles, full_tiles,
                                     tail_elems, buffer_free, input_gm);

  uint32_t issued = 0;
  for (uint32_t tile_index = core_id; tile_index < tiles;
       tile_index += core_count, ++issued) {
    const uint32_t buffer = issued % NBuffers;
    // issued ahead of the wait below, so this load overlaps this tile's compute
    issue_tile_load<Shape, NBuffers>(issued + NPrefetch, core_id, core_count,
                                     tiles, full_tiles, tail_elems, buffer_free,
                                     input_gm);
    wait_flag(PIPE_MTE2, PIPE_V, buffer_free[buffer]);
    const uint32_t slot_base = buffer * Shape::slot_stride;
#ifdef MXFP4_TQUANT
    tquant_passes<Shape>(slot_base + Offsets::input,
                         slot_base + Offsets::nibbles,
                         slot_base + Offsets::scales);
#else
    // name the UB regions once; inline casts are noise at every call site
    using B16 = __ubuf__ uint16_t *;
    B16 input_ub = (B16)(uintptr_t)(slot_base + Offsets::input);
    B16 scale_ub = (B16)(uintptr_t)(slot_base + Offsets::scales);
    B16 maxima_ub = (B16)(uintptr_t)Offsets::maxima;
    B16 packed_ub = (B16)(uintptr_t)Offsets::packed;
    B16 recips_ub = (B16)(uintptr_t)Offsets::reciprocal;
    __ubuf__ uint8_t *nibble_ub =
        (__ubuf__ uint8_t *)(uintptr_t)(slot_base + Offsets::nibbles);
    // rotate in place, then quantize the rotated tile without it ever leaving
    // UB
#ifndef FUSED_NO_ROTATE
    rotate<Shape>(input_ub);
#else
    // Diagnostic build: same kernel, same tiling, same UB layout and buffer
    // count -- only the butterfly removed. Comparing this against the quantizer
    // alone separates the butterfly's vector cost from the cost of fusing at
    // all (extra UB regions, so fewer buffers, so less overlap).
    (void)0;
#endif
#ifndef FUSED_ROTATE_ONLY
    block_abs_max<Shape>(input_ub, maxima_ub);
    compact_maxima<Shape>(maxima_ub, packed_ub);
    derive_scales<Shape>(packed_ub, recips_ub, scale_ub);
    pack_nibbles<Shape>(input_ub, recips_ub, nibble_ub);
#else
    // The other half of the fusion question. FUSED_NO_ROTATE keeps the
    // quantizer and drops the butterfly; this keeps the butterfly and drops the
    // quantizer, storing the rotated bf16 tile instead. Chained with the
    // standalone quantizer it is the UNFUSED reference: two launches, two
    // passes over HBM, 4 + 2.53 B/elem against the fused kernel's 2.53.
    //
    // Same tiling, UB layout and buffer count as the fused build, so the only
    // differences against it are the arithmetic skipped and the bytes stored.
    (void)scale_ub;
    (void)maxima_ub;
    (void)packed_ub;
    (void)recips_ub;
    (void)nibble_ub;
#endif
#endif
    set_flag(PIPE_V, PIPE_MTE3, buffer_free[buffer]);
    wait_flag(PIPE_V, PIPE_MTE3, buffer_free[buffer]);
#ifdef FUSED_ROTATE_ONLY
    // `nibble_gm` carries the rotated bf16 tile here and `scale_gm` is
    // untouched, so the launcher signature does not change. The harness
    // allocates 2K bytes per row for it, not K/2.
    if (tile_index == full_tiles) {
      move_tile<bfloat16_t, Shape::tile_elems, false, true>(
          tile_index, slot_base + Offsets::input, nibble_gm, tail_elems);
    } else {
      move_tile<bfloat16_t, Shape::tile_elems, false>(
          tile_index, slot_base + Offsets::input, nibble_gm);
    }
    (void)scale_gm;
#else
    if (tile_index == full_tiles) {
      move_tile<uint8_t, Shape::q_bytes, false, true>(
          tile_index, slot_base + Offsets::nibbles, nibble_gm, tail_elems / 2u);
      move_tile<uint8_t, Shape::scale_bytes, false, true>(
          tile_index, slot_base + Offsets::scales, scale_gm,
          tail_elems / MX_BLOCK);
    } else {
      move_tile<uint8_t, Shape::q_bytes, false>(
          tile_index, slot_base + Offsets::nibbles, nibble_gm);
      move_tile<uint8_t, Shape::scale_bytes, false>(
          tile_index, slot_base + Offsets::scales, scale_gm);
    }
#endif
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buffer]);
  }
  for (unsigned i = 0; i < NBuffers; ++i)  // drain
    wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
}
#endif  // __CCE_AICORE__ && __DAV_VEC__

template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
__global__ AICORE void mxfp4_quant(__gm__ void *input_gm,
                                   __gm__ void *nibble_gm,
                                   __gm__ void *scale_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  quant_tiles<K, Rows, NBuffers, NPrefetch>(input_gm, nibble_gm, scale_gm,
                                            batch);
#else
  (void)input_gm;
  (void)nibble_gm;
  (void)scale_gm;
  (void)batch;
#endif
}

#ifndef FUSED_INCLUDE_ONLY  // define to take the device code without hosts
// ---------------------------------------------------------------- entry points
// One .so serves every K: fold over SUPPORTED_K for the instantiation.
template <std::size_t... Idx>
inline void launch_for_k(uint32_t block_dim, void *stream, uint8_t *input,
                         uint8_t *nibbles, uint8_t *scales, uint32_t batch,
                         uint32_t k, std::index_sequence<Idx...>) {
  ((k == SUPPORTED_K[Idx]
        ? (void)(mxfp4_quant<SUPPORTED_K[Idx], RowsFor<SUPPORTED_K[Idx]>::value,
                             DEF_BUFFERS, DEF_PREFETCH>
                 <<<block_dim, nullptr, stream>>>(input, nibbles, scales,
                                                  batch))
        : (void)0),
   ...);
}

// An unsupported k is a silent no-op; the host validates
// (check_row_width).
extern "C" void call_hadamard_mxfp4_b32(uint32_t block_dim, void *stream,
                                        uint8_t *input, uint8_t *nibbles,
                                        uint8_t *scales, uint32_t batch,
                                        uint32_t k) {
  launch_for_k(block_dim, stream, input, nibbles, scales, batch, k,
               std::make_index_sequence<SUPPORTED_COUNT>{});
}

template <std::size_t... Idx>
inline uint32_t rows_for_k(uint32_t k, std::index_sequence<Idx...>) {
  uint32_t rows = 0;
  ((k == SUPPORTED_K[Idx] ? (void)(rows = RowsFor<SUPPORTED_K[Idx]>::value)
                          : (void)0),
   ...);
  return rows;  // 0 for an unsupported k
}

extern "C" uint32_t hadamard_mxfp4_b32_rows_for(uint32_t k) {
  return rows_for_k(k, std::make_index_sequence<SUPPORTED_COUNT>{});
}
#endif  // FUSED_INCLUDE_ONLY
