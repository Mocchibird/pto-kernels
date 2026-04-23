/**
 * Doubly-stochastic Sinkhorn normalization — minimal PTO demo (fp16, K=4).
 *
 * Mirrors DeepSeek TileKernels `sinkhorn_normalize_ref`:
 *
 *     x = softmax(x, -1) + eps
 *     x = x / (colsum(x) + eps)                        # first col normalize
 *     for _ in range(repeat - 1):
 *         x = x / (rowsum(x) + eps)                    # row normalize
 *         x = x / (colsum(x) + eps)                    # col normalize
 *
 * Parallelism: one 4×4 matrix per AIV core. Each core walks through
 * the batch with stride = total AIV count.
 *
 * Fp16 PTO tiles must have row-bytes that are a multiple of 32.
 * For fp16 that means the tile column dim must be a multiple of 16.
 * K=4 is smaller than that, so we pad to a 16×16 tile and keep the
 * 240 unused cells at 0 for the whole computation.
 */

#include <pto/pto-inst.hpp>

using namespace pto;

// ---- Problem constants ----
constexpr uint32_t K        = 4;   // matrix dimension
constexpr uint32_t TILE_DIM = 16;  // padded tile dim (32-byte row alignment for fp16)

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// ---- Tile type aliases ----
// A single 16×16 fp16 UB tile. Valid region is the top-left K×K.
using Matrix = Tile<TileType::Vec, half, TILE_DIM, TILE_DIM,
                    BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// A 1-D view over the same 256 halves (used when we only need elementwise ops).
using FlatView = Tile<TileType::Vec, half, 1, TILE_DIM * TILE_DIM,
                      BLayout::RowMajor, -1, -1>;

// TROWMAX / TROWSUM produce one scalar per ROW → a K×1 column vector.
using ColVec = Tile<TileType::Vec, half, TILE_DIM, 1,
                    BLayout::ColMajor, DYNAMIC, DYNAMIC>;

// TCOLSUM produces one scalar per COLUMN → a 1×K row vector.
using RowVec = Tile<TileType::Vec, half, 1, TILE_DIM,
                    BLayout::RowMajor, -1, -1>;

// ---- Global-memory view aliases ----
using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
using GmShape  = TileShape2D<half, DYNAMIC, DYNAMIC, Layout::ND>;
using GmTensor = GlobalTensor<half, GmShape, GmStride, Layout::ND>;


AICORE void sinkhornK4(__gm__ half *in, __gm__ half *out,
                       uint32_t num_matrices, uint32_t repeat, float eps) {
  // UB memory layout — three 16×16 fp16 slots (512 bytes each).
  constexpr unsigned MATRIX_BUF  = 0;
  constexpr unsigned SCRATCH_BUF = MATRIX_BUF  + TILE_DIM * TILE_DIM * sizeof(half);
  constexpr unsigned VECTOR_BUF  = SCRATCH_BUF + TILE_DIM * TILE_DIM * sizeof(half);

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t core_id   = get_block_idx() * get_subblockdim() + get_subblockid();
  const half     eps_h     = (half)eps;

  // Initial cross-pipe flags — prime them so the first wait_flag below succeeds.
  set_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);

  // Each AIV core processes every num_cores-th matrix in the batch.
  for (uint32_t idx = core_id; idx < num_matrices; idx += num_cores) {
    __gm__ half *in_gm  = in  + (size_t)idx * K * K;
    __gm__ half *out_gm = out + (size_t)idx * K * K;

    // ── Load the K×K input into the top-left of a 16×16 tile ─────────────
    // First zero the tile so the padding cells don't affect reductions.
    {
      FlatView all(1, TILE_DIM * TILE_DIM);
      TASSIGN(all, MATRIX_BUF);
      TEXPANDS(all, (half)0.f);
      pipe_barrier(PIPE_V);
    }

    Matrix mat(K, K);
    TASSIGN(mat, MATRIX_BUF);

    GmShape  gm_shape(K, K);
    GmStride gm_stride(K);
    GmTensor gm_in(in_gm, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // MTE2 waits for V to release the buffer
    TLOAD(mat, gm_in);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // tell V the TLOAD has landed
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Reusable views of the scratch + vector buffers.
    Matrix scratch(K, K);
    TASSIGN(scratch, SCRATCH_BUF);

    ColVec row_stat(K, 1);
    TASSIGN(row_stat, VECTOR_BUF);

    // ── Step 1: softmax along the last dim ───────────────────────────────
    //   mat = exp(mat - rowmax(mat))
    //   mat = mat / rowsum(mat)
    TROWMAX(row_stat, mat, scratch);
    pipe_barrier(PIPE_V);

    TROWEXPANDSUB(mat, mat, row_stat);
    pipe_barrier(PIPE_V);

    {
      FlatView flat(1, K * TILE_DIM);
      TASSIGN(flat, MATRIX_BUF);
      TEXP(flat, flat);
      pipe_barrier(PIPE_V);
    }

    TROWSUM(row_stat, mat, scratch);
    pipe_barrier(PIPE_V);

    TROWEXPANDDIV(mat, mat, row_stat);
    pipe_barrier(PIPE_V);

    // ── Step 2: mat += eps ──────────────────────────────────────────────
    {
      FlatView flat(1, K * TILE_DIM);
      TASSIGN(flat, MATRIX_BUF);
      TADDS(flat, flat, eps_h);
      pipe_barrier(PIPE_V);
    }

    // ── Step 3: first col-normalize  —  mat = mat / (colsum(mat) + eps) ──
    {
      RowVec col_stat(1, K);
      TASSIGN(col_stat, VECTOR_BUF);

      TCOLSUM(col_stat, mat, scratch, false);
      pipe_barrier(PIPE_V);

      TADDS(col_stat, col_stat, eps_h);
      pipe_barrier(PIPE_V);

      TCOLEXPANDDIV(mat, mat, col_stat);
      pipe_barrier(PIPE_V);
    }

    // ── Step 4: alternate row- and col-normalize, (repeat - 1) times ─────
    //   mat = mat / (rowsum(mat) + eps)
    //   mat = mat / (colsum(mat) + eps)
    for (uint32_t iter = 1; iter < repeat; ++iter) {
      TASSIGN(row_stat, VECTOR_BUF);

      TROWSUM(row_stat, mat, scratch);
      pipe_barrier(PIPE_V);

      TADDS(row_stat, row_stat, eps_h);
      pipe_barrier(PIPE_V);

      TROWEXPANDDIV(mat, mat, row_stat);
      pipe_barrier(PIPE_V);

      {
        RowVec col_stat(1, K);
        TASSIGN(col_stat, VECTOR_BUF);

        TCOLSUM(col_stat, mat, scratch, false);
        pipe_barrier(PIPE_V);

        TADDS(col_stat, col_stat, eps_h);
        pipe_barrier(PIPE_V);

        TCOLEXPANDDIV(mat, mat, col_stat);
        pipe_barrier(PIPE_V);
      }
    }

    // ── Store K×K back to GM ────────────────────────────────────────────
    GmTensor gm_out(out_gm, gm_shape, gm_stride);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(gm_out, mat);
    pipe_barrier(PIPE_ALL);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
  }

  // Drain pipelines before exit.
  wait_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);
}
#endif  // __CCE_AICORE__ == 220 && __DAV_C220_VEC__


// ---- C ABI ---------------------------------------------------------------

extern "C" __global__ AICORE void sinkhorn_k4_fp16(
    __gm__ uint8_t *input, __gm__ uint8_t *output,
    uint32_t num_matrices, uint32_t repeat, float eps) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  sinkhornK4((__gm__ half *)input, (__gm__ half *)output,
             num_matrices, repeat, eps);
#else
  (void)input;
  (void)output;
  (void)num_matrices;
  (void)repeat;
  (void)eps;
#endif
}

// Host-side launch. Ascend 910B runs 2 AIV cores per cube core.
extern "C" void call_sinkhorn(
    uint32_t cube_core_num, void *stream,
    uint8_t *input, uint8_t *output,
    uint32_t num_matrices, uint32_t repeat, float eps) {
  sinkhorn_k4_fp16<<<cube_core_num * 2, nullptr, stream>>>(
      input, output, num_matrices, repeat, eps);
}
