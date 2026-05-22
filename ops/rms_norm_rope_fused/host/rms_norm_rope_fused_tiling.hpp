#pragma once
#include <cstdint>

namespace harness {
namespace rms_norm_rope_fused {

struct RMSNormRoPETiling {
    int64_t B;            // batch size
    int64_t S;            // sequence length
    int64_t H;            // hidden dimension
    int64_t rotary_dim;   // RoPE dimension
    float eps = 1e-5f;

    // Tile: each block handles one (B,S) row
    int64_t tile_h = 256;        // elements per tile along H
    int64_t tile_rope = 64;      // elements per tile for rope portion
};

RMSNormRoPETiling compute_rms_norm_rope_tiling(int64_t B, int64_t S, int64_t H,
                                                int64_t rotary_dim, float eps = 1e-5f);

}  // namespace rms_norm_rope_fused
}  // namespace harness
