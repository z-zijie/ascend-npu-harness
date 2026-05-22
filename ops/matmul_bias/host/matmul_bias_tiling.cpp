#include "matmul_bias_tiling.hpp"
#include <algorithm>

namespace harness {
namespace matmul_bias {

MatmulBiasTiling compute_matmul_bias_tiling(int64_t M, int64_t K, int64_t N,
                                             bool has_bias) {
    MatmulBiasTiling t;
    t.M = M;
    t.K = K;
    t.N = N;
    t.has_bias = has_bias;

    // Block sizes for dav-2201: 32x32x32 uses 1024 elements
    // per output tile + 2048 elements for input tiles = ~3K elements
    // in UB, which is typical
    t.block_m = std::min(M, int64_t(32));
    t.block_n = std::min(N, int64_t(32));
    t.block_k = std::min(K, int64_t(32));

    if (t.block_m < 1) t.block_m = 1;
    if (t.block_n < 1) t.block_n = 1;
    if (t.block_k < 1) t.block_k = 1;

    return t;
}

}  // namespace matmul_bias
}  // namespace harness
