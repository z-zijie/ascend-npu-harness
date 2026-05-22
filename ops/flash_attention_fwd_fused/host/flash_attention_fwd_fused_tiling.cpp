#include "flash_attention_fwd_fused_tiling.hpp"
#include <algorithm>
#include <cmath>

namespace harness {
namespace flash_attention_fwd_fused {

FlashAttentionTiling compute_flash_attention_tiling(
    int64_t B, int64_t H,
    int64_t S_q, int64_t S_k,
    int64_t D, int64_t D_v,
    float scale, AttnMaskType mask_type) {

    FlashAttentionTiling t;
    t.B = B;
    t.H = H;
    t.S_q = S_q;
    t.S_k = S_k;
    t.D = D;
    t.D_v = D_v;
    t.scale = scale;
    t.mask_type = mask_type;

    // Tile sizes tuned for dav-2201 UB capacity (~192 KB)
    // block_q * block_k elements in score matrix (float)
    // plus block_k * block_d for K, block_q * block_d for Q
    //
    // Target: block_q * block_d <= 32 * 64 = 2048 floats = 8 KB per Q tile
    //         block_k * block_d <= 64 * 64 = 4096 floats = 16 KB per K/V tile
    //         block_q * block_k <= 32 * 64 = 2048 floats = 8 KB score tile
    // Total: ~32 KB per iteration, well within UB budget

    t.block_q = std::min(S_q, int64_t(32));
    t.block_k = std::min(S_k, int64_t(64));
    t.block_d = std::min(D, int64_t(64));

    // Adjust for small shapes
    if (S_q <= 16) t.block_q = S_q;
    if (S_k <= 16) t.block_k = S_k;
    if (D <= 16) t.block_d = D;

    // Ensure minimum of 1
    if (t.block_q < 1) t.block_q = 1;
    if (t.block_k < 1) t.block_k = 1;
    if (t.block_d < 1) t.block_d = 1;

    return t;
}

}  // namespace flash_attention_fwd_fused
}  // namespace harness
