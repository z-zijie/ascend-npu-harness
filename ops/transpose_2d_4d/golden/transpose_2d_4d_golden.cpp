#include "transpose_2d_4d_golden.hpp"
#include <cstring>

namespace harness {
namespace transpose_2d_4d {

template <typename T>
void transpose_2d_golden(const T* __restrict__ input,
                         T* __restrict__ output,
                         int64_t M, int64_t N) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            output[n * M + m] = input[m * N + n];
        }
    }
}

template <typename T>
void transpose_4d_nchw_to_nhwc_golden(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int64_t N, int64_t C, int64_t H, int64_t W) {
    // input: [N, C, H, W], output: [N, H, W, C]
    // in[n, c, h, w] -> out[n, h, w, c]
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    int64_t out_idx = ((n * H + h) * W + w) * C + c;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

template <typename T>
void transpose_4d_nhwc_to_nchw_golden(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int64_t N, int64_t C, int64_t H, int64_t W) {
    // input: [N, H, W, C], output: [N, C, H, W]
    // in[n, h, w, c] -> out[n, c, h, w]
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                for (int64_t c = 0; c < C; ++c) {
                    int64_t in_idx = ((n * H + h) * W + w) * C + c;
                    int64_t out_idx = ((n * C + c) * H + h) * W + w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// Template instantiations
template void transpose_2d_golden<float>(const float*, float*, int64_t, int64_t);
template void transpose_2d_golden<int32_t>(const int32_t*, int32_t*, int64_t, int64_t);
template void transpose_4d_nchw_to_nhwc_golden<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t);
template void transpose_4d_nchw_to_nhwc_golden<int32_t>(const int32_t*, int32_t*, int64_t, int64_t, int64_t, int64_t);
template void transpose_4d_nhwc_to_nchw_golden<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t);
template void transpose_4d_nhwc_to_nchw_golden<int32_t>(const int32_t*, int32_t*, int64_t, int64_t, int64_t, int64_t);

}  // namespace transpose_2d_4d
}  // namespace harness
