#include "rms_norm_rope_fused_tiling.hpp"
#include <algorithm>

namespace harness {
namespace rms_norm_rope_fused {

RMSNormRoPETiling compute_rms_norm_rope_tiling(int64_t B, int64_t S, int64_t H,
                                                int64_t rotary_dim, float eps) {
    RMSNormRoPETiling t;
    t.B = B;
    t.S = S;
    t.H = H;
    t.rotary_dim = rotary_dim;
    t.eps = eps;

    // Tile along H for efficient computation
    if (H <= 128) {
        t.tile_h = H;
    } else if (H <= 512) {
        t.tile_h = 256;
    } else {
        t.tile_h = 512;
    }

    // RoPE tile: handle rotary_dim elements at a time
    t.tile_rope = std::min(rotary_dim, int64_t(64));
    if (t.tile_rope < 2) t.tile_rope = 2;  // at least 1 pair

    return t;
}

}  // namespace rms_norm_rope_fused
}  // namespace harness
