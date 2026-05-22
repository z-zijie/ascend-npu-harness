#pragma once
#include <cstdint>
#include "flash_attention_fwd_fused_golden.hpp"

namespace harness {
namespace flash_attention_fwd_fused {

// Tiling configuration for flash attention forward.
// This controls how the attention computation is split across
// tiles for efficient memory access and compute on NPU.
struct FlashAttentionTiling {
    int64_t B;     // batch size
    int64_t H;     // number of heads
    int64_t S_q;   // query sequence length
    int64_t S_k;   // key sequence length
    int64_t D;     // head dimension (Q, K)
    int64_t D_v;   // value head dimension
    float scale;
    AttnMaskType mask_type;

    // Tile sizes
    int64_t block_b = 1;       // batch tile
    int64_t block_h = 1;       // head tile
    int64_t block_q = 32;      // query tile
    int64_t block_k = 64;      // key tile
    int64_t block_d = 64;      // dimension tile

    // Derive number of tiles
    int64_t num_q_tiles() const {
        return (S_q + block_q - 1) / block_q;
    }
    int64_t num_k_tiles() const {
        return (S_k + block_k - 1) / block_k;
    }
    int64_t num_d_tiles() const {
        return (D + block_d - 1) / block_d;
    }
};

// Compute optimal tiling for given attention shape
FlashAttentionTiling compute_flash_attention_tiling(
    int64_t B, int64_t H,
    int64_t S_q, int64_t S_k,
    int64_t D, int64_t D_v,
    float scale, AttnMaskType mask_type);

}  // namespace flash_attention_fwd_fused
}  // namespace harness
