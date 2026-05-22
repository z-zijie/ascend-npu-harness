#include "layer_norm_tiling.hpp"
#include <algorithm>

namespace harness {
namespace layer_norm {

LayerNormTiling compute_layer_norm_tiling(int64_t rows, int64_t last_dim, float eps) {
    LayerNormTiling t;
    t.rows = rows;
    t.last_dim = last_dim;
    t.eps = eps;

    // Tile along the last dimension for efficient reduction
    // Larger tiles reduce recursion but need more UB
    if (last_dim <= 128) {
        t.tile_dim = last_dim;
    } else if (last_dim <= 512) {
        t.tile_dim = 256;
    } else if (last_dim <= 2048) {
        t.tile_dim = 512;
    } else {
        t.tile_dim = 512;
    }
    return t;
}

}  // namespace layer_norm
}  // namespace harness
