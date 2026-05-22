#include "reduce_sum_lastdim_tiling.hpp"
#include <algorithm>

namespace harness {
namespace tiling {
namespace reduce_sum_lastdim {

TilingData compute_tiling(uint32_t num_batches, uint32_t M, uint32_t K) {
    TilingData td;
    td.num_batches = num_batches;
    td.M = M;
    td.K = K;
    td.total_rows = num_batches * M;
    td.tile_k = TILE_K;
    td.num_k_tiles = (K + TILE_K - 1) / TILE_K;
    td.tail_k = K % TILE_K;
    if (td.tail_k == 0 && K > 0) {
        td.tail_k = TILE_K;
    }
    return td;
}

}  // namespace reduce_sum_lastdim
}  // namespace tiling
}  // namespace harness
