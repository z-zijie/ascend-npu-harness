#pragma once
#include <cstdint>
#include "flash_attention_fwd_fused_golden.hpp"

namespace harness {
namespace flash_attention_fwd_fused {

// CPU runner — dispatches to golden
inline void run_flash_attention_fwd_fused_cpu(
    const float* Q, const float* K, const float* V,
    float* O,
    int64_t B, int64_t H,
    int64_t S_q, int64_t S_k,
    int64_t D, int64_t D_v,
    float scale,
    AttnMaskType mask_type) {
    flash_attention_fwd_golden(Q, K, V, O, B, H, S_q, S_k, D, D_v, scale, mask_type);
}

}  // namespace flash_attention_fwd_fused
}  // namespace harness
