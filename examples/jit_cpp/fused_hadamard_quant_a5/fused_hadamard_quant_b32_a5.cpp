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
// The difference from the companion fused_hadamard_quant_a5 is not just a
// pinned width. That one rotates a whole row, so K must be a power of two.
// Here the Hadamard is always 32 wide and a row is a sequence of independent
// 32-blocks, which decouples the rotation from the row width: K no longer has
// to be a power of two, so 4096, 5120 and 14336 are all instantiable.
//
// It does NOT make every multiple of 32 legal. The tile still has to divide
// into whole grains, which is what RowsFor solves; at the default TILE_ELEMS
// 76 of the 512 multiples of 32 up to 16384 admit a row count, and SUPPORTED_K
// instantiates 28 of them.
// A width that does not (11008, say) fails the Rows > 0 static_assert at
// compile time rather than misbehaving.
//
// It falls out of the tile being a flat run of Rows*K elements. Blocks are
// contiguous and 32 long, the butterfly window is 256 = eight blocks, and
// blocks are independent -- so one window covers eight of them and never has to
// care whether they came from the same row.
//
// Also: the MXFP4 group is 32 and the Hadamard block is 32, so a scale covers
// exactly one rotated block. No reshaping, and no group straddling a rotation.
#include "fused_hadamard_quant_common.hpp"

// Row widths with an instantiation. A 32-wide rotation puts no power-of-two
// constraint on the row, so 4096- and 14336-style widths are in. A new width
// must still satisfy RowsFor (see above) and be added to the jit helper.
constexpr unsigned SUPPORTED_K[] = {32,   64,   96,   128,  192,  256,   512,
                                    768,  896,  1024, 1152, 1280, 1408,  1536,
                                    1664, 1792, 2048, 2560, 2816, 3072,  3584,
                                    4096, 5120, 6144, 7168, 8192, 14336, 16384};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_K) / sizeof(SUPPORTED_K[0]);
constexpr unsigned HAD_BLOCK = 32;  // the Hadamard block, == MX_BLOCK

#ifdef __CCE_AICORE__
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

#ifdef __DAV_VEC__
// --- the rotation ------------------------------------------------------------
// One sweep: deinterleave-load a window, add/sub, and store the halves back
// concatenated. Registers are vector_u16 because vlds/vsts are bit-width ops;
// the arithmetic type is chosen by reference cast, which is how bf16 costs
// nothing here. All loads precede all stores, which the comma fold guarantees
// by evaluating left to right -- required, not stylistic, since a store would
// otherwise clobber a window a later load still needs.

template <typename Shape, unsigned Rotations, std::size_t... Slot>
inline AICORE void sweep(__ubuf__ uint16_t *tile, uint32_t base, MaskReg all,
                         HadRegs &even, HadRegs &odd, HadRegs &sum,
                         HadRegs &diff, std::index_sequence<Slot...>) {
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
  // dead by then. vdintlv costs about 20x a vadd and there are Rotations of
  // them per slot against five arithmetic ops, which makes this fixup 13.8%
  // of the kernel -- measured, see the store note at the top of the file.
  if constexpr (Rotations >= 1) {
    (vdintlv(even[Slot], odd[Slot], sum[Slot], diff[Slot]), ...);
  }
  if constexpr (Rotations >= 2) {
    (vdintlv(sum[Slot], diff[Slot], even[Slot], odd[Slot]), ...);
  }
  if constexpr (Rotations >= 3) {
    (vdintlv(even[Slot], odd[Slot], sum[Slot], diff[Slot]), ...);
  }
  HadRegs &lo = (Rotations % 2 == 1) ? even : sum;
  HadRegs &hi = (Rotations % 2 == 1) ? odd : diff;
  (vsts(lo[Slot], tile + base + Slot / ch * g + Slot % ch * ln, 0, NORM_B16,
        all),
   ...);
  (vsts(hi[Slot], tile + base + Slot / ch * g + up + Slot % ch * ln, 0,
        NORM_B16, all),
   ...);
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
    // Step by a literal 1 with the stride folded into base: the loop analyser
    // only verifies a tripcount for a literal step, and 1 divides any bound, so
    // had_iters may be template-dependent.
    for (uint16_t stage = 0; stage < (uint16_t)plain; ++stage) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::had_iters; ++iter)
        sweep<Shape, 0>(tile, (uint32_t)iter * Shape::sweep_stride, all, even,
                        odd, sum, diff, slots);
      mem_bar(VST_VLD);
    }
    if constexpr (Shape::rotations > 0) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::had_iters; ++iter)
        sweep<Shape, Shape::rotations>(tile,
                                       (uint32_t)iter * Shape::sweep_stride,
                                       all, even, odd, sum, diff, slots);
      mem_bar(VST_VLD);
    }
  }
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

// The pipeline: each core walks a strided subset of the tiles, keeping Prefetch
// loads in flight so DMA and the vector pipe overlap.
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
// A device function rather than the kernel body, so a caller that wants the
// pipeline over a sub-range can reach it directly. The kernel below is the
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
__global__ AICORE void fused_hadamard_mxfp4_b32(__gm__ void *input_gm,
                                                __gm__ void *nibble_gm,
                                                __gm__ void *scale_gm,
                                                uint32_t batch) {
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
        ? (void)(fused_hadamard_mxfp4_b32<SUPPORTED_K[Idx],
                                          RowsFor<SUPPORTED_K[Idx]>::value,
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
