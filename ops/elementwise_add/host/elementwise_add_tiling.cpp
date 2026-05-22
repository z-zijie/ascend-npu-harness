#include "elementwise_add_tiling.hpp"
#include <algorithm>

namespace harness {
namespace tiling {
namespace elementwise_add {

TilingData compute_tiling(uint32_t total_elements) {
    TilingData td;
    td.total_elements = total_elements;
    td.tile_elements = TILE_ELEMENTS;
    td.num_tiles = (total_elements + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
    td.tail_elements = total_elements % TILE_ELEMENTS;
    if (td.tail_elements == 0 && total_elements > 0) {
        td.tail_elements = TILE_ELEMENTS;
    }
    return td;
}

}  // namespace elementwise_add
}  // namespace tiling
}  // namespace harness
