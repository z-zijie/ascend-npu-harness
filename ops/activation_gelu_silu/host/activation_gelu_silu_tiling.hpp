#pragma once
#include <cstdint>

namespace harness {
namespace tiling {
namespace activation_gelu_silu {

// Tiling parameters for activation_gelu_silu kernel (dav-2201)
// Strategy: 1D tiling along the flattened element count.
constexpr uint32_t TILE_ELEMENTS = 1024;

// activation_type: 0 = GELU, 1 = SiLU
struct TilingData {
    uint32_t total_elements;
    uint32_t tile_elements;
    uint32_t num_tiles;
    uint32_t tail_elements;
    uint32_t activation_type;  // 0=gelu, 1=silu
};

TilingData compute_tiling(uint32_t total_elements, uint32_t activation_type);

}  // namespace activation_gelu_silu
}  // namespace tiling
}  // namespace harness
