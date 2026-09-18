/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "utils.h"

extern "C" {

void pto_launch_kda_kkt(uint32_t blockDim, void* stream, void* k_ptr,
                        void* g_cs_ptr, void* beta_ptr, void* mask_ptr,
                        void* L_out_ptr, void* cu_seqlens, int64_t batch_size,
                        int64_t seq_len, int64_t total_tokens);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Runs the kda_kkt kernel: within-chunk gated attention matrix for KDA.
 *
 * Computes per chunk, for each head h and row r > c (strictly lower-tri):
 *   L[r, c] = beta[r] * sum_d k[r,d] * k[c,d] * exp(min(g_cs[r,d]-g_cs[c,d],
 * 0))
 *
 * Tensor layouts (head-major):
 *   K    — [H, total_tokens, D]  fp16
 *   G_cs — [H, total_tokens, D]  fp32  within-chunk cumulative gate sum
 *   Beta — [H, total_tokens]     fp16  post-sigmoid scalar in (0, 1)
 *
 * @param K          fp16 key tensor           [H, total_tokens, D]  head-major
 * @param G_cs       fp32 cumulative gate sum  [H, total_tokens, D]  head-major
 * @param Beta       fp16 beta tensor          [H, total_tokens]     head-major
 * @param batch_size Number of sequences in the batch.
 * @param seq_len    Tokens per sequence (fixed-length path).
 *                   Ignored when cu_seqlens is provided.
 * @param cu_seqlens Optional int32 cumulative sequence lengths [batch_size+1].
 *                   Pass at::zeros({1}) (default) to use the fixed-length path.
 * @return at::Tensor fp16 output L [total_tokens, H, C] (BSND layout),
 *         strictly-lower-triangular within each chunk; upper-tri entries are 0.
 */
at::Tensor run_kda_kkt(const at::Tensor& K, const at::Tensor& G_cs,
                       const at::Tensor& Beta, int64_t batch_size,
                       int64_t seq_len,
                       const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "kda_kkt: tensors must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "kda_kkt: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(G_cs.scalar_type() == at::kFloat,
              "kda_kkt: G_cs must be fp32, got ", G_cs.scalar_type());
  TORCH_CHECK(Beta.scalar_type() == at::kHalf,
              "kda_kkt: Beta must be fp16, got ", Beta.scalar_type());
  TORCH_CHECK(K.is_contiguous(), "kda_kkt: K must be contiguous");
  TORCH_CHECK(G_cs.is_contiguous(), "kda_kkt: G_cs must be contiguous");
  TORCH_CHECK(Beta.is_contiguous(), "kda_kkt: Beta must be contiguous");

  // K: [H, total_tokens, D]; G_cs: [H, total_tokens, D]; Beta: [H,
  // total_tokens]
  const int64_t num_heads = K.size(0);
  const int64_t total_tokens = K.size(1);
  // KDA_KKT_C compile-time constant — matches kernel default build
  constexpr int64_t CHUNK_C = 128;

  // Work items are HalfChunk x ColBlock blocks of L per (chunk, head), and
  // the kernel now chooses ColBlock itself from the item count against
  // get_block_num().  So do not cap block_dim here: capping it would feed
  // that choice a smaller concurrency than the device has and push it to a
  // narrower block than the shape wants.  This is a Vec-only kernel, so the
  // lanes to fill are the AIV cores, and the kernel's flat enumeration idles
  // any lane left without an item.
  const uint32_t block_dim = GetNumVectorCores();

  // The kernel derives the strict-lower mask as a computed step and never
  // reads mask_ptr, which stays in the signature only for ABI stability.
  // Building a [C, C] tril here cost an at::ones and an at::tril launch on
  // every call for a buffer nobody loads.
  const at::Tensor mask = at::empty({0}, G_cs.options());

  // Output L [total_tokens, H, C] fp16, BSND layout.
  // Zero-initialised: the kernel only writes strict-lower-tri entries.
  const at::Tensor L_out =
      at::zeros({total_tokens, num_heads, CHUNK_C}, K.options());

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(kda_kkt, block_dim, K, G_cs, Beta, mask, L_out,
                  cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return L_out;
}

}  // namespace pto_isa_ops
