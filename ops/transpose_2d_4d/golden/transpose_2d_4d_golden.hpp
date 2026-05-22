#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace harness {
namespace transpose_2d_4d {

// 2D transpose: [M, N] -> [N, M]
// out[n][m] = in[m][n]
template <typename T>
void transpose_2d_golden(const T* __restrict__ input,
                         T* __restrict__ output,
                         int64_t M, int64_t N);

// 4D NCHW -> NHWC: [N, C, H, W] -> [N, H, W, C]
template <typename T>
void transpose_4d_nchw_to_nhwc_golden(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int64_t N, int64_t C, int64_t H, int64_t W);

// 4D NHWC -> NCHW: [N, H, W, C] -> [N, C, H, W]
template <typename T>
void transpose_4d_nhwc_to_nchw_golden(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int64_t N, int64_t C, int64_t H, int64_t W);

extern template void transpose_2d_golden<float>(const float*, float*, int64_t, int64_t);
extern template void transpose_2d_golden<int32_t>(const int32_t*, int32_t*, int64_t, int64_t);
extern template void transpose_4d_nchw_to_nhwc_golden<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t);
extern template void transpose_4d_nchw_to_nhwc_golden<int32_t>(const int32_t*, int32_t*, int64_t, int64_t, int64_t, int64_t);
extern template void transpose_4d_nhwc_to_nchw_golden<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t);
extern template void transpose_4d_nhwc_to_nchw_golden<int32_t>(const int32_t*, int32_t*, int64_t, int64_t, int64_t, int64_t);

}  // namespace transpose_2d_4d
}  // namespace harness
