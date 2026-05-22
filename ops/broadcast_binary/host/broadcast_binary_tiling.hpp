#pragma once
#include <cstdint>

namespace harness {
namespace tiling {
namespace broadcast_binary {

// Tiling parameters for broadcast_binary kernel (dav-2201)
// Strategy: 2D tiling over H and W spatial dimensions.
// C is the inner vectorization axis (broadcast along spatial tile).
constexpr uint32_t TILE_H = 32;
constexpr uint32_t TILE_W = 32;

struct TilingData {
    uint32_t B;
    uint32_t C;
    uint32_t H;
    uint32_t W;
    uint32_t y_batch;       // 1 or B
    uint32_t tile_h;
    uint32_t tile_w;
    uint32_t num_tiles_h;
    uint32_t num_tiles_w;
    uint32_t tail_h;
    uint32_t tail_w;
};

TilingData compute_tiling(uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                          uint32_t y_batch);

}  // namespace broadcast_binary
}  // namespace tiling
}  // namespace harness
