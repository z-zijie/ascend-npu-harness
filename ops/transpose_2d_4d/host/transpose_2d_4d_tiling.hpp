#pragma once
#include <cstdint>

namespace harness {
namespace transpose_2d_4d {

// Tiling parameters for transpose kernel
struct Transpose2DTiling {
    int64_t M;
    int64_t N;
    int64_t block_m = 16;
    int64_t block_n = 16;
};

struct Transpose4DTiling {
    int64_t N;  // batch
    int64_t C;  // channels
    int64_t H;  // height
    int64_t W;  // width
    int64_t block_c = 16;
    int64_t block_hw = 16;
};

enum class TransposeMode {
    Transpose2D = 0,
    NCHWToNHWC = 1,
    NHWCToNCHW = 2,
};

// Compute 2D transpose tiling
Transpose2DTiling compute_transpose_2d_tiling(int64_t M, int64_t N);

// Compute 4D transpose tiling
Transpose4DTiling compute_transpose_4d_tiling(int64_t N, int64_t C, int64_t H, int64_t W,
                                               TransposeMode mode);

}  // namespace transpose_2d_4d
}  // namespace harness
