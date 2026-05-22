#include "activation_gelu_silu_tiling.hpp"
#include <algorithm>

namespace harness {
namespace tiling {
namespace activation_gelu_silu {

TilingData compute_tiling(uint32_t total_elements, uint32_t activation_type) {
    TilingData td;
    td.total_elements = total_elements;
    td.tile_elements = TILE_ELEMENTS;
    td.num_tiles = (total_elements + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
    td.tail_elements = total_elements % TILE_ELEMENTS;
    if (td.tail_elements == 0 && total_elements > 0) {
        td.tail_elements = TILE_ELEMENTS;
    }
    td.activation_type = activation_type;
    return td;
}

}  // namespace activation_gelu_silu
}  // namespace tiling
}  // namespace harness
