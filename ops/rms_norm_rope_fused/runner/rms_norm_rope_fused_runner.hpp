#pragma once
#include <cstdint>
#include "rms_norm_rope_fused_golden.hpp"

namespace harness {
namespace rms_norm_rope_fused {

// CPU runner — dispatches to golden
inline void run_rms_norm_rope_fused_cpu(const float* x,
                                         const float* weight,
                                         const float* cos,
                                         const float* sin,
                                         float* out,
                                         int64_t B, int64_t S, int64_t H,
                                         int64_t rotary_dim, float eps) {
    rms_norm_rope_fused_golden(x, weight, cos, sin, out, B, S, H, rotary_dim, eps);
}

}  // namespace rms_norm_rope_fused
}  // namespace harness
