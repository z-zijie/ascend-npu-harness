#pragma once
#include <cstdint>

namespace harness {
namespace tiling {
namespace reduce_sum_lastdim {

// Tiling parameters for reduce_sum_lastdim kernel (dav-2201)
// Strategy: split along non-reduced dimensions (B * M) across cores.
// Each core reduces its rows along K.
// Vector width: 16 elements (fp32 accumulation)
constexpr uint32_t TILE_K = 256;  // Elements per reduction chunk

struct TilingData {
    uint32_t num_batches;   // B (1 for 2D, B for 3D)
    uint32_t M;             // rows per batch
    uint32_t K;             // reduction dimension
    uint32_t total_rows;    // = num_batches * M
    uint32_t tile_k;        // chunk size along K
    uint32_t num_k_tiles;   // = ceil(K / tile_k)
    uint32_t tail_k;        // = K % tile_k (or tile_k if 0)
};

TilingData compute_tiling(uint32_t num_batches, uint32_t M, uint32_t K);

}  // namespace reduce_sum_lastdim
}  // namespace tiling
}  // namespace harness
