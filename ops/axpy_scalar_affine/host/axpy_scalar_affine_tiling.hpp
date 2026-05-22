#pragma once
#include <cstdint>

namespace harness {
namespace tiling {
namespace axpy_scalar_affine {

// Tiling parameters for axpy_scalar_affine kernel (dav-2201)
// Strategy: 1D tiling along the flattened element count.
constexpr uint32_t TILE_ELEMENTS = 1024;

struct TilingData {
    uint32_t total_elements;
    uint32_t tile_elements;
    uint32_t num_tiles;
    uint32_t tail_elements;
    float alpha;
    float beta;
};

TilingData compute_tiling(uint32_t total_elements, float alpha, float beta);

}  // namespace axpy_scalar_affine
}  // namespace tiling
}  // namespace harness
