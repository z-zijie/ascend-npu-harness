#pragma once
#include <cstdint>

namespace harness {
namespace layer_norm {

struct LayerNormTiling {
    int64_t rows;          // M (2D) or B*S (3D)
    int64_t last_dim;      // K (2D) or H (3D)
    int64_t tile_dim = 256;  // elements per tile along last_dim
    float eps = 1e-5f;
};

LayerNormTiling compute_layer_norm_tiling(int64_t rows, int64_t last_dim, float eps = 1e-5f);

}  // namespace layer_norm
}  // namespace harness
