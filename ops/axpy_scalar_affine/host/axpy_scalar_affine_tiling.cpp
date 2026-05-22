#include "axpy_scalar_affine_tiling.hpp"
#include <algorithm>

namespace harness {
namespace tiling {
namespace axpy_scalar_affine {

TilingData compute_tiling(uint32_t total_elements, float alpha, float beta) {
    TilingData td;
    td.total_elements = total_elements;
    td.tile_elements = TILE_ELEMENTS;
    td.num_tiles = (total_elements + TILE_ELEMENTS - 1) / TILE_ELEMENTS;
    td.tail_elements = total_elements % TILE_ELEMENTS;
    if (td.tail_elements == 0 && total_elements > 0) {
        td.tail_elements = TILE_ELEMENTS;
    }
    td.alpha = alpha;
    td.beta = beta;
    return td;
}

}  // namespace axpy_scalar_affine
}  // namespace tiling
}  // namespace harness
