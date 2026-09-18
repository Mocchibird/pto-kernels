// ============================================================================
// kernel_kda_kkt.cpp — Within-chunk gated attention matrix for KDA (numerically
// stable)
//
// Mathematical operation (per chunk of C tokens, per head h):
//   L[r,c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(g_cs[r,d] - g_cs[c,d])
//            for r > c (strictly lower-tri), else 0
//
// STABILITY: Kimi KDA gates (g = -exp(A_log)*softplus(...)) are unbounded; the
//   within-chunk cumulative gate g_cs can reach ~-500.  The previous factorized
//   form  A_eff=k*exp(g_cs), B_eff=k*exp(-g_cs), L=A_eff@B_eff^T  computes
//   exp(-g_cs)=exp(+500) which overflows fp32 (max e^88) -> inf -> inf*0 = NaN.
//
//   This kernel instead computes exp(g_cs[r]-g_cs[c]) as a DIFFERENCE, never
//   the product of two separate exponentials.  For the kept (lower-tri) entries
//   r>c, g_cs[r] <= g_cs[c] (g_cs monotone decreasing within a chunk) so the
//   argument is <= 0 and exp(.) <= 1 — always finite.  The upper-tri entries
//   are never stored and TROWSUM keeps rows independent, so they need no
//   clamp to stay out of a kept row.  No pivot, no saturation, exact.
//
// IMPLEMENTATION: a work item is one (chunk, head, column block).  The kernel
//   picks the column block width itself, scoring each candidate width by
//   ceil(items / get_block_num()) * columns_per_item, so a short sequence
//   splits into enough items to fill every lane; ties go to the widest block,
//   which pays the fewest per-item prologues.  Each item walks its columns c
//   and computes its rows' column-c of L via a per-column elementwise
//   reduction:
//
//     diff[r,d] = g_cs[my_row r, d] - g_cs[c, d]   (TCOLEXPANDSUB, per-dim c)
//     t[r,d]    = exp(diff) * kb[c,d] * k[my_row r,d]
//                                                  (TEXP, TCOLEXPANDMUL, TMUL)
//     L[my_row r, c] = sum_d t[r,d]                (TROWSUM)
//
//   where kb = k * beta is built once per chunk, so no per-column beta
//   multiply is left.  The strict-lower mask is a computed step on the
//   leading rows, not a gathered [C, C] tensor.
//   Then it stores the strict-lower rows (global_row > c) to L_out.  This is a
//   Vec-only kernel; the Cube pass only participates in the entry/exit
//   barriers. (A GEMM-accelerated off-diagonal path is a future optimization.)
//
// Inputs (all on GM, head-major [HV, total_tokens, K]):
//   k       [HV, total_tokens, K]  float16  — keys
//   g_cs    [HV, total_tokens, K]  float32  — within-chunk cumulative gate sum
//   beta    [HV, total_tokens]     float16  — post-sigmoid beta in (0, 1)
//   mask    [C, C]                 float32  — (unused; kept for ABI stability)
//   ws_in   [block_dim*2, 2*C, K]  float32  — (unused; kept for ABI stability)
//   ws_out  [block_dim*2, C, C]    float32  — (unused; kept for ABI stability)
//   L_out   [total_tokens, HV, C]  float16  — strictly-lower-tri L (BSND)
//
// Template parameters: KDA_KKT_H = HV, KDA_KKT_D = K, KDA_KKT_C = C
// ============================================================================

#include "kernel_utils.h"

using namespace pto;
using kernel_utils::PipeBarrierVec;

#ifndef KDA_KKT_H
#define KDA_KKT_H 4
#endif

#ifndef KDA_KKT_D
#define KDA_KKT_D 128
#endif

#ifndef KDA_KKT_C
#define KDA_KKT_C 128
#endif

#ifdef __CCE_AICORE__

template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, RV,
                       CV, pto::SLayout::NoneBox, 512, P>;

// Column-vector tiles ([R,1]) must be ColMajor: RowMajor NoneBox requires the
// column byte-width to be 32-byte aligned, which width-1 tiles fail.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor, RV,
                       CV, pto::SLayout::NoneBox, 512>;
#endif

template <int32_t NumHeads, int32_t KDim, int32_t ChunkSize>
AICORE inline void kda_kkt_kernel(__gm__ half* k_ptr, __gm__ float* g_cs_ptr,
                                  __gm__ half* beta_ptr, __gm__ float* mask_ptr,
                                  __gm__ half* L_out_ptr,
                                  __gm__ int32_t* cu_seqlens,
                                  int64_t batch_size, int64_t seq_len,
                                  int64_t total_tokens) {
  using pto::Stride;
  // This is a Vec-only kernel (no Cube branch), so bisheng compiles it as a
  // pure-AIV kernel where get_subblockid() is always 0.  Enumerate the launched
  // AIV lanes flatly (same idiom as kernel_swiglu / kernel_tri_inv_col_sweep)
  // and fold the per-chunk row half into the work item so both halves of every
  // (seq, head) are covered regardless of how many lanes are launched.
  const uint32_t num_lanes = get_block_num() * get_subblockdim();
  const uint32_t lane = get_block_idx() * get_subblockdim() + get_subblockid();

  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t KTC = ((KDim + 7) / 8) * 8;

  const int64_t num_seqs = batch_size;
  // Work items are (chunk, head, row_half): half=0 -> rows [0, C/2), half=1
  // -> rows [C/2, C).  Chunks are independent — a chunk's L depends only on
  // that chunk's own rows — so enumerating them flatly gives num_chunks times
  // more items and keeps every launched lane busy.
  int64_t total_chunks = 0;
  if (cu_seqlens != nullptr) {
    for (int64_t s = 0; s < num_seqs; ++s) {
      const int64_t sl = static_cast<int64_t>(cu_seqlens[s + 1]) -
                         static_cast<int64_t>(cu_seqlens[s]);
      total_chunks += (sl + ChunkSize - 1) / ChunkSize;
    }
  } else {
    total_chunks = num_seqs * ((seq_len + ChunkSize - 1) / ChunkSize);
  }
  // Work items are HalfChunk x ColBlock blocks of L.  The lower row half
  // only reaches column HalfChunk-1 so it contributes HalfChunk/ColBlock of
  // them; the upper half spans the chunk and contributes ChunkSize/ColBlock.
  // Blocks entirely above the diagonal are never emitted, and every item is
  // the same size, so the makespan is one block.
  //
  // ColBlock is picked here rather than fixed, because the right width
  // depends on how many items the shape yields against how many run at
  // once: a short sequence with wide blocks leaves lanes idle, while a long
  // one with narrow blocks pays the per-item prologue (the row g_cs/k/beta
  // loads and mykb) too many times.  Cost of a candidate is (rounds it
  // needs) x (columns per item); ties go to the widest block, which pays
  // that prologue fewest times.
  //
  // The divisor is get_block_num(), not num_lanes.  Work is still handed out
  // across all num_lanes lanes, but measured makespan scales with
  // items/get_block_num() on both parts: on A2A3 num_lanes is twice
  // get_block_num() (two Vec sub-blocks per block) and yet doubling the item
  // count from 48 to 96 doubles the time, so the sub-blocks do not add
  // concurrency here.  Using num_lanes made this pick ColBlock 32 at T=512
  // on A2A3 and cost 4.7%.
  //
  // Every lane runs the same arithmetic on the same inputs, so they all
  // choose the same width and agree on the decode.
  static_assert(HalfChunk % 8 == 0,
                "Fix: ChunkSize/2 must be a multiple of 8 for the ColBlock "
                "search to reach every candidate width.");
  const int64_t units = total_chunks * NumHeads;
  int32_t ColBlock = HalfChunk;
  int64_t best_cost = -1;
  for (int32_t cb = HalfChunk; cb >= 8; cb >>= 1) {
    const int64_t slots = (ChunkSize / cb) + (HalfChunk / cb);
    const int64_t items = units * slots;
    const int64_t concurrency = static_cast<int64_t>(get_block_num());
    const int64_t rounds = (items + concurrency - 1) / concurrency;
    const int64_t cost = rounds * static_cast<int64_t>(cb);
    if (best_cost < 0 || cost < best_cost) {
      best_cost = cost;
      ColBlock = cb;
    }
  }
  const int32_t LowerBlocks = HalfChunk / ColBlock;
  const int32_t UpperBlocks = ChunkSize / ColBlock;
  const int32_t SlotsPerChunkHead = LowerBlocks + UpperBlocks;
  const int64_t total_work = units * static_cast<int64_t>(SlotsPerChunkHead);

  // ── GM type aliases (head-major [HV, T, K]) ──────────────────────────────
  using GmShapeDyn = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmFloatK = GlobalTensor<float, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
  using GmHalfK = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, KDim, 1>>;
  using GmHalf1 = GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, 1, 1>>;
  // L_out is [total_tokens, HV, C].  We store one column c of L for a strided
  // set of global rows: a [1, store_rows] view whose "columns" step by one
  // global row (distance NumHeads*ChunkSize in L_out), col stride below.
  // Store one column c of L for a strided set of global rows: the row (token)
  // dim steps by NumHeads*ChunkSize, the single column is contiguous (stride
  // 1).
  using GmHalfLoutCol =
      GlobalTensor<half, GmShapeDyn, Stride<1, 1, 1, NumHeads * ChunkSize, 1>>;
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── UB layout (per vid) ──────────────────────────────────────────────────
  constexpr int32_t MYG_ADDR = 0;  // [HalfChunk, K] fp32
  constexpr int32_t MYK_ADDR =
      MYG_ADDR + HalfChunk * KTC * 4;  // [HalfChunk, K] fp32
  constexpr int32_t DIFF_ADDR =
      MYK_ADDR + HalfChunk * KTC * 4;  // [HalfChunk, K] fp32 (diff/t)
  constexpr int32_t TMP_ADDR =
      DIFF_ADDR + HalfChunk * KTC * 4;  // [HalfChunk, K] fp32 (rowsum tmp)
  constexpr int32_t MYKH_ADDR =
      TMP_ADDR + HalfChunk * KTC * 4;  // [HalfChunk, K] fp16 (k staging)
  constexpr int32_t GC_ADDR =
      MYKH_ADDR + HalfChunk * KTC * 2;             // [1, K] fp32 (column g_c)
  constexpr int32_t KC_ADDR = GC_ADDR + KTC * 4;   // [1, K] fp32 (column k_c)
  constexpr int32_t KCH_ADDR = KC_ADDR + KTC * 4;  // [1, K] fp16 (k_c staging)
  constexpr int32_t COL_ADDR =
      KCH_ADDR + KTC * 2;  // [HalfChunk, 16] fp32 (colsum, RowMajor padded)
  constexpr int32_t COLH_ADDR =
      COL_ADDR +
      HalfChunk * 16 * 4;  // [HalfChunk, 16] fp16 (padded store, RowMajor)
  constexpr int32_t BETA_ADDR =
      COLH_ADDR + HalfChunk * 16 * 2;  // [1, HalfChunk] fp32 (beta)
  constexpr int32_t BETAH_ADDR =
      BETA_ADDR + HalfChunk * 4;  // [1, HalfChunk] fp16 (beta staging)
  constexpr int32_t MSKC_ADDR =
      BETAH_ADDR + HalfChunk * 2;  // [1, HalfChunk] fp32 (mask col)
  // k scaled by beta, built once per chunk.  myk itself has to stay
  // unscaled because the column loop reads its rows as column vectors.
  constexpr int32_t MYKB_ADDR =
      MSKC_ADDR + HalfChunk * 4;  // [HalfChunk, K] fp32 (k * beta)

  for (int64_t pid = static_cast<int64_t>(lane); pid < total_work;
       pid += static_cast<int64_t>(num_lanes)) {
    // Decode work item: (global chunk index, head_idx, block slot).
    const int32_t slot = static_cast<int32_t>(pid % SlotsPerChunkHead);
    const int64_t hc = pid / SlotsPerChunkHead;
    const int32_t head_idx = static_cast<int32_t>(hc % NumHeads);
    int64_t ci = hc / NumHeads;
    const int32_t row_half = slot < LowerBlocks ? 0 : 1;
    const int32_t col_block = slot < LowerBlocks ? slot : slot - LowerBlocks;
    const int32_t my_off = row_half * HalfChunk;
    const int32_t col_begin = col_block * ColBlock;

    // Resolve the global chunk index to its sequence.  The varlen scan is
    // over num_seqs scalars, which is negligible next to the chunk body.
    int64_t bos = 0, slen = 0;
    if (cu_seqlens != nullptr) {
      for (int64_t s = 0; s < num_seqs; ++s) {
        bos = static_cast<int64_t>(cu_seqlens[s]);
        slen = static_cast<int64_t>(cu_seqlens[s + 1]) - bos;
        const int64_t nc = (slen + ChunkSize - 1) / ChunkSize;
        if (ci < nc) break;
        ci -= nc;
      }
    } else {
      const int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
      bos = (ci / chunks_per_seq) * seq_len;
      slen = seq_len;
      ci = ci % chunks_per_seq;
    }

    const int64_t chunk_start = ci * ChunkSize;
    const int64_t remaining = slen - chunk_start;
    const int32_t valid_rows =
        static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);

    // This vid's row range within the chunk: [my_off, my_off + my_rows).
    const int32_t my_rows_raw = valid_rows - my_off;
    const int32_t my_rows = my_rows_raw > HalfChunk ? HalfChunk : my_rows_raw;
    if (my_rows <= 0) continue;  // no rows for this vid in this chunk

    // Columns this item must cover: c in [col_begin, col_end).  A row r is
    // kept for column c only if global_row(r) > c, so the largest column any
    // of my rows touches is (my_off + my_rows - 1); the slot also caps it at
    // its own column block.
    const int32_t col_cap = col_begin + ColBlock;
    const int32_t rows_cap = my_off + my_rows;
    const int32_t col_end = rows_cap < col_cap ? rows_cap : col_cap;
    if (col_begin >= col_end) continue;  // block is entirely above the diagonal

    const int64_t hbase = static_cast<int64_t>(head_idx) * total_tokens * KDim;
    const int64_t my_first =
        bos + chunk_start + my_off;  // global row index of my row 0

    // ── Load my rows' g_cs (fp32) and k (fp16 -> fp32) ───────────────
    {
      GmShapeDyn gs;
      gs.shape[3] = my_rows;
      gs.shape[4] = KDim;
      GmFloatK g_gm(g_cs_ptr + hbase + my_first * KDim, gs);
      UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero> g_ld(
          my_rows, KDim);
      TASSIGN(g_ld, MYG_ADDR);
      TLOAD(g_ld, g_gm);
    }
    {
      GmShapeDyn tensor;
      tensor.shape[3] = my_rows;
      tensor.shape[4] = KDim;
      GmHalfK k_gm(k_ptr + hbase + my_first * KDim, tensor);
      UbND<half, HalfChunk, KTC, DYNAMIC, DYNAMIC, PadValue::Zero> k_ld(my_rows,
                                                                        KDim);
      TASSIGN(k_ld, MYKH_ADDR);
      TLOAD(k_ld, k_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    {
      UbND<half, HalfChunk, KTC, DYNAMIC, DYNAMIC> k_h(my_rows, KDim);
      TASSIGN(k_h, MYKH_ADDR);
      UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> k_f(my_rows, KDim);
      TASSIGN(k_f, MYK_ADDR);
      TCVT(k_f, k_h, pto::RoundMode::CAST_NONE);
      PipeBarrierVec();
    }
    // ── Load my rows' beta (fp16 -> fp32) as a [1, my_rows] row, then
    //    re-view as a [my_rows, 1] column for the per-row scale. ───────
    {
      GmShapeDyn gs;
      gs.shape[3] = 1;
      gs.shape[4] = my_rows;
      GmHalf1 b_gm(
          beta_ptr + static_cast<int64_t>(head_idx) * total_tokens + my_first,
          gs);
      UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC> b_ld(1, my_rows);
      TASSIGN(b_ld, BETAH_ADDR);
      TLOAD(b_ld, b_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    {
      UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC> b_h(1, my_rows);
      TASSIGN(b_h, BETAH_ADDR);
      UbND<float, 1, HalfChunk, DYNAMIC, DYNAMIC> b_f(1, my_rows);
      TASSIGN(b_f, BETA_ADDR);
      TCVT(b_f, b_h, pto::RoundMode::CAST_NONE);
      PipeBarrierVec();
    }

    UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> myg(my_rows, KDim);
    TASSIGN(myg, MYG_ADDR);
    UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> myk(my_rows, KDim);
    TASSIGN(myk, MYK_ADDR);
    UbDN<float, HalfChunk, 1, DYNAMIC, DYNAMIC> beta_col(my_rows, 1);
    TASSIGN(beta_col, BETA_ADDR);
    // beta[r] multiplies every term of row r's sum, so fold it into k once
    // per chunk instead of scaling the [HalfChunk, KDim] product tile on
    // every one of the up-to-128 columns.
    UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> mykb(my_rows, KDim);
    TASSIGN(mykb, MYKB_ADDR);
    TROWEXPANDMUL(mykb, myk, beta_col);
    PipeBarrierVec();

    // ── Column loop ──────────────────────────────────────────────────
    for (int32_t c = col_begin; c < col_end; ++c) {
      // Column c's g_cs and k.  Columns in [my_off, my_off + my_rows) are my
      // own rows, whose g_cs (myg) and fp32 k (myk) are already in UB, so
      // point at row c - my_off instead of re-reading GM and re-casting.
      // The lower row half never leaves that range, and the upper half only
      // does for its first my_off columns.
      int32_t gc_addr = GC_ADDR;
      int32_t kc_addr = KC_ADDR;
      if (c >= my_off && c - my_off < my_rows) {
        const int32_t r = c - my_off;
        gc_addr = MYG_ADDR + r * KTC * 4;
        kc_addr = MYK_ADDR + r * KTC * 4;
      } else {
        const int64_t col_off = hbase + (bos + chunk_start + c) * KDim;
        {
          GmShapeDyn gs;
          gs.shape[3] = 1;
          gs.shape[4] = KDim;
          GmFloatK gc_gm(g_cs_ptr + col_off, gs);
          UbND<float, 1, KTC, 1, KTC> gc_ld;
          TASSIGN(gc_ld, GC_ADDR);
          TLOAD(gc_ld, gc_gm);
          GmHalfK kc_gm(k_ptr + col_off, gs);
          UbND<half, 1, KTC, 1, KTC> kc_ld;
          TASSIGN(kc_ld, KCH_ADDR);
          TLOAD(kc_ld, kc_gm);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          UbND<half, 1, KTC, 1, KTC> kc_h;
          TASSIGN(kc_h, KCH_ADDR);
          UbND<float, 1, KTC, 1, KTC> kc_f;
          TASSIGN(kc_f, KC_ADDR);
          TCVT(kc_f, kc_h, pto::RoundMode::CAST_NONE);
          PipeBarrierVec();
        }
      }

      UbND<float, 1, KTC, 1, KTC> gc;
      TASSIGN(gc, gc_addr);
      UbND<float, 1, KTC, 1, KTC> kc;
      TASSIGN(kc, kc_addr);
      UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> diff(my_rows, KDim);
      TASSIGN(diff, DIFF_ADDR);
      UbND<float, HalfChunk, KTC, DYNAMIC, DYNAMIC> tmp(my_rows, KDim);
      TASSIGN(tmp, TMP_ADDR);
      UbND<float, HalfChunk, 16, DYNAMIC, DYNAMIC> colsum(my_rows, 1);
      TASSIGN(colsum, COL_ADDR);

      // diff[r,d] = g_cs[my r, d] - g_cs[c, d]
      TCOLEXPANDSUB(diff, myg, gc);
      PipeBarrierVec();
      // No clamp before the exp.  The argument is > 0 only where
      // g_cs[my r] > g_cs[c], i.e. only for rows above the diagonal
      // (my_off + r <= c) -- and those are exactly the rows the strict-lower
      // step below overwrites with zero, so an inf or NaN there never
      // reaches L_out.  Rows are independent through TROWSUM, so a poisoned
      // row cannot contaminate a kept one.  A short chunk is no exception:
      // every tile here is sized to my_rows, the live row count, so there are
      // no zero-padded rows carrying g_cs = 0 into the exp -- the trap
      // kda_chunk_o does fall into, because it pads its tiles out to HalfC.
      // Saves a full-tile TMINS and its barrier per column.
      TEXP(diff, diff);
      PipeBarrierVec();
      // *= k[c,d]  (per-dim broadcast), then *= k[my r, d]
      TCOLEXPANDMUL(diff, diff, kc);
      PipeBarrierVec();
      TMUL(diff, diff, mykb);
      PipeBarrierVec();
      // colsum[r] = beta[r] * sum_d diff[r,d]   (unmasked)
      TROWSUM(colsum, diff, tmp);
      PipeBarrierVec();

      // Strict-lower mask.  The kept rows for column c are exactly those
      // with my_off + r > c, so the mask is a step: zero the leading
      // c - my_off + 1 rows of colsum and leave the rest.  Writing those
      // rows with TEXPANDS replaces a strided [my_rows,1] gather from GM
      // plus its MTE2->V flag pair and a TMUL, and for the upper half it
      // is skipped entirely while c < my_off.
      const int32_t zero_rows_raw = c - my_off + 1;
      const int32_t zero_rows =
          zero_rows_raw < 0
              ? 0
              : (zero_rows_raw > my_rows ? my_rows : zero_rows_raw);
      if (zero_rows > 0) {
        UbND<float, HalfChunk, 16, DYNAMIC, DYNAMIC> mk0(zero_rows, 1);
        TASSIGN(mk0, COL_ADDR);
        TEXPANDS(mk0, 0.0f);
        PipeBarrierVec();
      }

      // cvt colsum -> fp16 into a padded RowMajor [my_rows, 1] tile and
      // store column c of L for the strided set of tokens (row dim steps
      // by NumHeads*ChunkSize, the single column is contiguous).
      UbND<half, HalfChunk, 16, DYNAMIC, DYNAMIC> col_h(my_rows, 1);
      TASSIGN(col_h, COLH_ADDR);
      TCVT(col_h, colsum, pto::RoundMode::CAST_NONE);
      PipeBarrierVec();

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        const int64_t l_off =
            my_first * static_cast<int64_t>(NumHeads) * ChunkSize +
            static_cast<int64_t>(head_idx) * ChunkSize + c;
        GmShapeDyn gs;
        gs.shape[3] = my_rows;
        gs.shape[4] = 1;
        GmHalfLoutCol l_gm(L_out_ptr + l_off, gs);
        TSTORE(l_gm, col_h);
      }
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // Drain all pipes before the next column iteration so its gc/kc
      // loads (MTE2) cannot race the current column's still-draining
      // Vec reads of the same UB slots.
      pipe_barrier(PIPE_ALL);
    }
  }
}

// ── Device entry point
// ────────────────────────────────────────────────────────
extern "C" __global__ AICORE void kda_kkt(
    __gm__ uint8_t* k_ptr, __gm__ uint8_t* g_cs_ptr, __gm__ uint8_t* beta_ptr,
    __gm__ uint8_t* mask_ptr, __gm__ uint8_t* L_out_ptr,
    __gm__ uint8_t* cu_seqlens, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens) {
#if defined(__DAV_VEC__)
  kda_kkt_kernel<KDA_KKT_H, KDA_KKT_D, KDA_KKT_C>(
      reinterpret_cast<__gm__ half*>(k_ptr),
      reinterpret_cast<__gm__ float*>(g_cs_ptr),
      reinterpret_cast<__gm__ half*>(beta_ptr),
      reinterpret_cast<__gm__ float*>(mask_ptr),
      reinterpret_cast<__gm__ half*>(L_out_ptr),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens), batch_size, seq_len,
      total_tokens);
#endif
}

// Host-callable launch shims: the `<<<>>>` syntax is only
// understood by the kernel compiler, so the launch lives here
// rather than in the host wrappers under csrc/host/.
extern "C" void pto_launch_kda_kkt(uint32_t blockDim, void* stream, void* k_ptr,
                                   void* g_cs_ptr, void* beta_ptr,
                                   void* mask_ptr, void* L_out_ptr,
                                   void* cu_seqlens, int64_t batch_size,
                                   int64_t seq_len, int64_t total_tokens) {
  kda_kkt<<<blockDim, nullptr, stream>>>(
      (__gm__ uint8_t*)k_ptr, (__gm__ uint8_t*)g_cs_ptr,
      (__gm__ uint8_t*)beta_ptr, (__gm__ uint8_t*)mask_ptr,
      (__gm__ uint8_t*)L_out_ptr, (__gm__ uint8_t*)cu_seqlens, batch_size,
      seq_len, total_tokens);
}
