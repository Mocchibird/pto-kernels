// fast_hadamard_128_dintlv_a5 — N=128 Walsh-Hadamard via deinterleave-load
// butterfly (the memory-bound technique of fast_hadamard_256_a5, scaled to
// N=128). Each of the log2(N) stages performs the even/odd split on the MTE2
// load (vlds DINTLV_B16) and the concat-halves recombine on the MTE3 store
// (vsts to [0 : N/2) and [N/2 : N)), leaving only vadd/vsub on the vector pipe.
// In-place on a UB tile. Unnormalized (x @ Sylvester(128)); a final-stage vmuls
// by 1/sqrt(128) would make it a drop-in for the register/cube kernels.
//
// A 128-element row deinterleaves into 64 even + 64 odd lanes, so the active
// lane count HALF_LANES = HAD_N / 2. Compares against a plain copy-floor of the
// same tiling (copy128_dintlv).

#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef HAD_N
#define HAD_N 128
#endif
#ifndef HAD_LOG2N
#define HAD_LOG2N 7
#endif
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 128
#endif
#ifndef ROWS_PER_UNROLL
#define ROWS_PER_UNROLL 8
#endif
#ifndef PIPELINE_BUFFERS
#define PIPELINE_BUFFERS 4
#endif
#ifndef PREFETCH_TILES
#define PREFETCH_TILES 2
#endif

constexpr uint32_t HALF_LANES = HAD_N / 2;                       // even/odd halves of one row
constexpr uint32_t ELEMENTS_PER_TILE = ROWS_PER_TILE * HAD_N;    // fp16 elements per UB tile
constexpr uint32_t TILE_BYTES = ELEMENTS_PER_TILE * sizeof(half);

/// @brief Round a byte count up to the next 512-byte boundary.
constexpr uint32_t alignUp512(uint32_t bytes) { return (bytes + 511u) & ~511u; }

constexpr uint32_t TILE_STRIDE_BYTES = alignUp512(TILE_BYTES);

/// @brief Byte offset of pipeline buffer `index` within UB (runtime index, so a
///        macro rather than a constexpr fn — the device compiler only evaluates
///        constexpr functions at compile time).
#define BUFFER_BASE(index) ((uint32_t)(index) * TILE_STRIDE_BYTES)

static_assert(PIPELINE_BUFFERS * TILE_STRIDE_BYTES <= 192u * 1024u, "UB overflow (>192 KB)");
static_assert(PIPELINE_BUFFERS <= 8, "at most 8 event IDs per pipe-pair");

#ifdef __DAV_VEC__

/// @brief One constant-geometry deinterleave-load butterfly stage, unrolled
///        ROWS_PER_UNROLL rows at a time.
/// @param workBuffer  UB byte pointer to the tile being transformed in place.
/// @param rowCount    Number of length-N rows in the tile.
///
/// Per unrolled row the load (vlds DINTLV_B16) splits the row into even/odd
/// halves, vadd/vsub form the sums/diffs, and the store writes sums to the low
/// half and diffs to the high half. The even/odd split and the concat-halves
/// recombine ride the load/store units, so only vadd/vsub touch the vector
/// pipe -> memory-bound.
__tf__ static AICORE void hadamardButterfly128(__ubuf__ half *workBuffer, uint32_t rowCount) {
  __VEC_SCOPE__ {
    uint32_t activeLanes = HALF_LANES;
    MaskReg halfMask = CreatePredicate<half>(activeLanes);

    vector_f16 even0, even1, even2, even3, even4, even5, even6, even7;
    vector_f16 odd0, odd1, odd2, odd3, odd4, odd5, odd6, odd7;
    vector_f16 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
    vector_f16 diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;

    for (uint16_t stage = 0; stage < (uint16_t)HAD_LOG2N; ++stage) {
      for (uint16_t row = 0; row < (uint16_t)rowCount; row += ROWS_PER_UNROLL) {
        const uint32_t rowBase = (uint32_t)row * HAD_N;

        // Deinterleaving load: even lanes -> evenN, odd lanes -> oddN.
        vlds(even0, odd0, workBuffer + rowBase + 0 * HAD_N, 0, DINTLV_B16);
        vlds(even1, odd1, workBuffer + rowBase + 1 * HAD_N, 0, DINTLV_B16);
        vlds(even2, odd2, workBuffer + rowBase + 2 * HAD_N, 0, DINTLV_B16);
        vlds(even3, odd3, workBuffer + rowBase + 3 * HAD_N, 0, DINTLV_B16);
        vlds(even4, odd4, workBuffer + rowBase + 4 * HAD_N, 0, DINTLV_B16);
        vlds(even5, odd5, workBuffer + rowBase + 5 * HAD_N, 0, DINTLV_B16);
        vlds(even6, odd6, workBuffer + rowBase + 6 * HAD_N, 0, DINTLV_B16);
        vlds(even7, odd7, workBuffer + rowBase + 7 * HAD_N, 0, DINTLV_B16);

        // Butterfly sums.
        vadd(sum0, even0, odd0, halfMask);
        vadd(sum1, even1, odd1, halfMask);
        vadd(sum2, even2, odd2, halfMask);
        vadd(sum3, even3, odd3, halfMask);
        vadd(sum4, even4, odd4, halfMask);
        vadd(sum5, even5, odd5, halfMask);
        vadd(sum6, even6, odd6, halfMask);
        vadd(sum7, even7, odd7, halfMask);

        // Butterfly differences.
        vsub(diff0, even0, odd0, halfMask);
        vsub(diff1, even1, odd1, halfMask);
        vsub(diff2, even2, odd2, halfMask);
        vsub(diff3, even3, odd3, halfMask);
        vsub(diff4, even4, odd4, halfMask);
        vsub(diff5, even5, odd5, halfMask);
        vsub(diff6, even6, odd6, halfMask);
        vsub(diff7, even7, odd7, halfMask);

        // Concat-halves recombine: sums to the low half, diffs to the high half.
        vsts(sum0, workBuffer + rowBase + 0 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff0, workBuffer + rowBase + 0 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum1, workBuffer + rowBase + 1 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff1, workBuffer + rowBase + 1 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum2, workBuffer + rowBase + 2 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff2, workBuffer + rowBase + 2 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum3, workBuffer + rowBase + 3 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff3, workBuffer + rowBase + 3 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum4, workBuffer + rowBase + 4 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff4, workBuffer + rowBase + 4 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum5, workBuffer + rowBase + 5 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff5, workBuffer + rowBase + 5 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum6, workBuffer + rowBase + 6 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff6, workBuffer + rowBase + 6 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
        vsts(sum7, workBuffer + rowBase + 7 * HAD_N, 0, NORM_B16, halfMask);
        vsts(diff7, workBuffer + rowBase + 7 * HAD_N + HALF_LANES, 0, NORM_B16, halfMask);
      }
      mem_bar(VST_VLD);
    }
  }
}

#endif  // __DAV_VEC__

/// @brief N=128 deinterleave-load Walsh-Hadamard transform, in place on x_gm.
/// @param x_gm  Global fp16 buffer of `batch` rows of HAD_N.
/// @param batch Number of rows; must be a multiple of ROWS_PER_TILE.
__global__ AICORE void fastHadamard128Dintlv(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  set_mask_norm();
  set_vector_mask(-1, -1);

  using TileShape = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using TileStride = pto::Stride<1, 1, 1, ELEMENTS_PER_TILE, 1>;
  using FlatTile = Tile<TileType::Vec, half, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, 1, ELEMENTS_PER_TILE>;

  const event_t bufferEvent[8] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
                                  EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7};
  const uint32_t coreId = get_block_idx();
  const uint32_t coreCount = get_block_num();
  const uint32_t tileCount = batch / ROWS_PER_TILE;

  // Issue the async GM->UB load for this core's step-k tile into buffer k % PIPELINE_BUFFERS.
#define ISSUE_LOAD(step)                                                       \
  do {                                                                         \
    uint32_t loadTile = coreId + (uint32_t)(step)*coreCount;                   \
    if (loadTile < tileCount) {                                                \
      const int buffer = (uint32_t)(step) % PIPELINE_BUFFERS;                  \
      wait_flag(PIPE_MTE3, PIPE_MTE2, bufferEvent[buffer]);                    \
      FlatTile loadView;                                                       \
      TASSIGN(loadView, BUFFER_BASE(buffer));                                   \
      GlobalTensor<half, TileShape, TileStride> loadSrc(                       \
          (__gm__ half *)x_gm + (uint64_t)loadTile * ELEMENTS_PER_TILE, TileShape()); \
      TLOAD(loadView, loadSrc);                                                \
      set_flag(PIPE_MTE2, PIPE_V, bufferEvent[buffer]);                        \
    }                                                                          \
  } while (0)

  for (int buffer = 0; buffer < PIPELINE_BUFFERS; ++buffer)
    set_flag(PIPE_MTE3, PIPE_MTE2, bufferEvent[buffer]);
  for (uint32_t step = 0; step < (uint32_t)PREFETCH_TILES; ++step)
    ISSUE_LOAD(step);

  uint32_t step = 0;
  for (uint32_t tile = coreId; tile < tileCount; tile += coreCount, ++step) {
    const int buffer = step % PIPELINE_BUFFERS;

    ISSUE_LOAD(step + PREFETCH_TILES);
    wait_flag(PIPE_MTE2, PIPE_V, bufferEvent[buffer]);

    hadamardButterfly128((__ubuf__ half *)(uintptr_t)BUFFER_BASE(buffer), ROWS_PER_TILE);

    set_flag(PIPE_V, PIPE_MTE3, bufferEvent[buffer]);
    wait_flag(PIPE_V, PIPE_MTE3, bufferEvent[buffer]);

    FlatTile storeView;
    TASSIGN(storeView, BUFFER_BASE(buffer));
    GlobalTensor<half, TileShape, TileStride> storeDst(
        (__gm__ half *)x_gm + (uint64_t)tile * ELEMENTS_PER_TILE, TileShape());
    TSTORE(storeDst, storeView);
    set_flag(PIPE_MTE3, PIPE_MTE2, bufferEvent[buffer]);
  }

  for (int buffer = 0; buffer < PIPELINE_BUFFERS; ++buffer)
    wait_flag(PIPE_MTE3, PIPE_MTE2, bufferEvent[buffer]);
#undef ISSUE_LOAD
#else
  (void)x_gm;
  (void)batch;
#endif
}

/// @brief Copy-floor reference: same tiling / double-buffered GM->UB->GM bounce
///        with no compute, to bound the achievable DMA bandwidth.
__global__ AICORE void copy128Dintlv(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using TileShape = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using TileStride = pto::Stride<1, 1, 1, ELEMENTS_PER_TILE, 1>;
  using FlatTile = Tile<TileType::Vec, half, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, 1, ELEMENTS_PER_TILE>;

  const event_t bufferEvent[2] = {EVENT_ID0, EVENT_ID1};
  const uint32_t coreId = get_block_idx();
  const uint32_t coreCount = get_block_num();
  const uint32_t tileCount = batch / ROWS_PER_TILE;

  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

  uint32_t step = 0;
  for (uint32_t tile = coreId; tile < tileCount; tile += coreCount, ++step) {
    const int buffer = step & 1;
    const uint64_t offset = (uint64_t)tile * ELEMENTS_PER_TILE;

    wait_flag(PIPE_MTE3, PIPE_MTE2, bufferEvent[buffer]);
    FlatTile view;
    TASSIGN(view, BUFFER_BASE(buffer));
    GlobalTensor<half, TileShape, TileStride> gm((__gm__ half *)x_gm + offset, TileShape());
    TLOAD(view, gm);
    set_flag(PIPE_MTE2, PIPE_MTE3, bufferEvent[buffer]);
    wait_flag(PIPE_MTE2, PIPE_MTE3, bufferEvent[buffer]);
    TSTORE(gm, view);
    set_flag(PIPE_MTE3, PIPE_MTE2, bufferEvent[buffer]);
  }

  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#else
  (void)x_gm;
  (void)batch;
#endif
}

extern "C" void call_hadamard128_dintlv(uint32_t block_dim, void *stream, uint8_t *x, uint32_t batch) {
  fastHadamard128Dintlv<<<block_dim, nullptr, stream>>>(x, batch);
}

extern "C" void call_copy128_dintlv(uint32_t block_dim, void *stream, uint8_t *x, uint32_t batch) {
  copy128Dintlv<<<block_dim, nullptr, stream>>>(x, batch);
}
