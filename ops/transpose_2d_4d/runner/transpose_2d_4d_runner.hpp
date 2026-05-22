#pragma once
#include <cstdint>
#include <cstring>
#include "transpose_2d_4d_golden.hpp"
#include "transpose_2d_4d_tiling.hpp"

namespace harness {
namespace transpose_2d_4d {

template <typename T>
void run_transpose_2d_cpu(const T* input, T* output,
                          int64_t M, int64_t N) {
    transpose_2d_golden(input, output, M, N);
}

template <typename T>
void run_transpose_4d_nchw_to_nhwc_cpu(const T* input, T* output,
                                        int64_t N, int64_t C, int64_t H, int64_t W) {
    transpose_4d_nchw_to_nhwc_golden(input, output, N, C, H, W);
}

template <typename T>
void run_transpose_4d_nhwc_to_nchw_cpu(const T* input, T* output,
                                        int64_t N, int64_t C, int64_t H, int64_t W) {
    transpose_4d_nhwc_to_nchw_golden(input, output, N, C, H, W);
}

}  // namespace transpose_2d_4d
}  // namespace harness
