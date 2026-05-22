#pragma once
#include <cstdint>

namespace harness {
namespace tiling {
namespace elementwise_add {

// Tiling parameters for elementwise_add kernel (dav-2201)
// Strategy: 1D tiling along the flattened element count.
// Vector width: 16 elements (fp32)
// UB capacity: ~192KB per core
constexpr uint32_t TILE_ELEMENTS = 1024;

struct TilingData {
    uint32_t total_elements;
    uint32_t tile_elements;
    uint32_t num_tiles;
    uint32_t tail_elements;
};

// Compute tiling data from total element count
TilingData compute_tiling(uint32_t total_elements);

}  // namespace elementwise_add
}  // namespace tiling
}  // namespace harness
