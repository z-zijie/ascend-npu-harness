#include "transpose_2d_4d_tiling.hpp"
#include <algorithm>

namespace harness {
namespace transpose_2d_4d {

Transpose2DTiling compute_transpose_2d_tiling(int64_t M, int64_t N) {
    Transpose2DTiling t;
    t.M = M;
    t.N = N;
    // Use block sizes that fit in local memory
    // 16x16 = 256 elements per block, fits well in UB
    t.block_m = std::min(M, int64_t(16));
    t.block_n = std::min(N, int64_t(16));
    // Ensure at least 1
    if (t.block_m < 1) t.block_m = 1;
    if (t.block_n < 1) t.block_n = 1;
    return t;
}

Transpose4DTiling compute_transpose_4d_tiling(int64_t N, int64_t C, int64_t H, int64_t W,
                                               TransposeMode mode) {
    Transpose4DTiling t;
    t.N = N;
    t.C = C;
    t.H = H;
    t.W = W;
    t.block_c = std::min(C, int64_t(16));
    t.block_hw = std::min(H * W, int64_t(16));
    if (t.block_c < 1) t.block_c = 1;
    if (t.block_hw < 1) t.block_hw = 1;
    return t;
}

}  // namespace transpose_2d_4d
}  // namespace harness
