#pragma once
#include <cstdint>

namespace harness {
namespace flash_attention_fwd_fused {

// Flash Attention Forward — Fused (CPU Golden)
//
// Computes scaled dot-product attention with online softmax for numerical stability.
//
// Shapes:
//   Q: [B, H, S_q, D]
//   K: [B, H, S_k, D]
//   V: [B, H, S_k, D_v]
//   O: [B, H, S_q, D_v]  (default D_v = D)
//
// Algorithm:
//   For each (b, h, q_idx):
//     m = -inf, l = 0, acc = zeros(D_v)
//     For each k_tile in range(0, S_k, tile_k):
//       scores = Q[b,h,q_idx,:] @ K[b,h,k_tile:k_tile+tile_k,:]^T * scale
//       Apply mask (none or causal)
//       m_new = max(m, max(scores))
//       l = l * exp(m - m_new) + sum(exp(scores - m_new))
//       acc = acc * exp(m - m_new) + exp(scores - m_new) @ V[b,h,k_tile:k_tile+tile_k,:]
//       m = m_new
//     O[b,h,q_idx,:] = acc / l
//
// Parameters:
//   mask_type: 0 = no mask, 1 = causal (valid only when S_q == S_k)
//   scale: 1/sqrt(D) by default, can be customized
//   D_v: value head dimension (defaults to D)
//
// All inputs/outputs are float (accumulation in float64 for online softmax).

enum class AttnMaskType {
    None = 0,
    Causal = 1,
};

void flash_attention_fwd_golden(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ O,
                                 int64_t B, int64_t H,
                                 int64_t S_q, int64_t S_k,
                                 int64_t D, int64_t D_v,
                                 float scale,
                                 AttnMaskType mask_type);

}  // namespace flash_attention_fwd_fused
}  // namespace harness
