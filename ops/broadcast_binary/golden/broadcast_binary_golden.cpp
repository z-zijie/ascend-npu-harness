#include "broadcast_binary_golden.hpp"
#include <stdexcept>

namespace harness {
namespace golden {
namespace broadcast_binary {

void compute(const float* x,
             const float* y,
             const float* bias,
             float* z,
             int64_t B, int64_t C, int64_t H, int64_t W,
             int64_t y_batch) {
    if (x == nullptr || y == nullptr || bias == nullptr || z == nullptr) {
        throw std::invalid_argument("broadcast_binary golden: null pointer");
    }
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) {
        throw std::invalid_argument("broadcast_binary golden: invalid dimensions");
    }
    if (y_batch != 1 && y_batch != B) {
        throw std::invalid_argument("broadcast_binary golden: y_batch must be 1 or B");
    }

    // x is contiguous NCHW: x[b*C*H*W + c*H*W + h*W + w]
    // y is [y_batch, C, 1, 1]: y_yb*C + c
    // bias is [C]: bias[c]

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t y_b_idx = (y_batch == 1) ? 0 : b;
            float y_val = y[y_b_idx * C + c];
            float bias_val = bias[c];
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((b * C + c) * H + h) * W + w;
                    z[idx] = x[idx] * y_val + bias_val;
                }
            }
        }
    }
}

}  // namespace broadcast_binary
}  // namespace golden
}  // namespace harness
