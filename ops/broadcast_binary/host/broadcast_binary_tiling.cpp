#include "broadcast_binary_tiling.hpp"
#include <algorithm>

namespace harness {
namespace tiling {
namespace broadcast_binary {

TilingData compute_tiling(uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                          uint32_t y_batch) {
    TilingData td;
    td.B = B;
    td.C = C;
    td.H = H;
    td.W = W;
    td.y_batch = y_batch;
    td.tile_h = TILE_H;
    td.tile_w = TILE_W;
    td.num_tiles_h = (H + TILE_H - 1) / TILE_H;
    td.num_tiles_w = (W + TILE_W - 1) / TILE_W;
    td.tail_h = H % TILE_H;
    if (td.tail_h == 0 && H > 0) td.tail_h = TILE_H;
    td.tail_w = W % TILE_W;
    if (td.tail_w == 0 && W > 0) td.tail_w = TILE_W;
    return td;
}

}  // namespace broadcast_binary
}  // namespace tiling
}  // namespace harness
