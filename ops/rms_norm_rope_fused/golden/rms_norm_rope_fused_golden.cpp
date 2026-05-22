#include "rms_norm_rope_fused_golden.hpp"
#include <cmath>
#include <cstring>

namespace harness {
namespace rms_norm_rope_fused {

void rms_norm_rope_fused_golden(const float* __restrict__ x,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ cos,
                                 const float* __restrict__ sin,
                                 float* __restrict__ out,
                                 int64_t B, int64_t S, int64_t H,
                                 int64_t rotary_dim,
                                 float eps) {
    int64_t rotary_half = rotary_dim / 2;
    int64_t num_elements = B * S * H;

    // Copy output to temp first (RMSNorm, then RoPE)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            const float* x_bs = x + (b * S + s) * H;
            float* out_bs = out + (b * S + s) * H;

            // Compute RMSNorm: rms = sqrt(mean(x^2) + eps)
            double sum_sq = 0.0;
            for (int64_t h = 0; h < H; ++h) {
                double val = static_cast<double>(x_bs[h]);
                sum_sq += val * val;
            }
            float rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(H) + static_cast<double>(eps)));

            // Apply RMSNorm: y = x / rms * weight
            // (store intermediate in out, then RoPE overwrites first rotary_dim entries)
            for (int64_t h = 0; h < H; ++h) {
                out_bs[h] = (x_bs[h] / rms) * weight[h];
            }

            // Apply RoPE on first rotary_dim dimensions (in-place within out_bs)
            const float* cos_s = cos + s * rotary_half;
            const float* sin_s = sin + s * rotary_half;
            for (int64_t d = 0; d < rotary_half; ++d) {
                int64_t even = 2 * d;
                int64_t odd = even + 1;
                float y_even = out_bs[even];
                float y_odd = out_bs[odd];
                float c = cos_s[d];
                float z = sin_s[d];
                out_bs[even] = y_even * c - y_odd * z;
                out_bs[odd]  = y_even * z + y_odd * c;
            }
            // Dimensions [rotary_dim, H) remain as RMSNorm output (already set above)
        }
    }
}

}  // namespace rms_norm_rope_fused
}  // namespace harness
