#pragma once
#include <cstdint>

namespace harness {
namespace rms_norm_rope_fused {

// Fused RMSNorm + RoPE operator.
//
// Step 1 - RMSNorm:
//   rms = sqrt(mean(x^2) + eps)
//   y = x / rms * weight   (for all H dimensions)
//
// Step 2 - RoPE on first rotary_dim dimensions:
//   For d = 0,2,...,rotary_dim-2 (even indices):
//     cos_idx = s * (rotary_dim/2) + d/2
//     out[d]   = y[d] * cos[cos_idx] - y[d+1] * sin[cos_idx]
//     out[d+1] = y[d] * sin[cos_idx] + y[d+1] * cos[cos_idx]
//   For d = rotary_dim ... H-1:
//     out[d] = y[d]  (pass through after RMSNorm only)
//
// Input:
//   x:       [B, S, H]
//   weight:  [H]
//   cos:     [S, rotary_dim/2]
//   sin:     [S, rotary_dim/2]
//   eps:     scalar
//   rotary_dim: <= H, must be even
//
// Output:
//   out: [B, S, H]

void rms_norm_rope_fused_golden(const float* __restrict__ x,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ cos,
                                 const float* __restrict__ sin,
                                 float* __restrict__ out,
                                 int64_t B, int64_t S, int64_t H,
                                 int64_t rotary_dim,
                                 float eps);

}  // namespace rms_norm_rope_fused
}  // namespace harness
