#pragma once
#include <cstdint>

namespace harness {
namespace matmul_bias {

struct MatmulBiasTiling {
    int64_t M;       // rows of A/C
    int64_t K;       // inner dimension
    int64_t N;       // columns of B/C
    bool has_bias;   // whether bias is present

    // Tile sizes for AscendC kernel
    int64_t block_m = 32;
    int64_t block_n = 32;
    int64_t block_k = 32;
};

MatmulBiasTiling compute_matmul_bias_tiling(int64_t M, int64_t K, int64_t N,
                                             bool has_bias);

}  // namespace matmul_bias
}  // namespace harness
