#include "layer_norm_golden.hpp"
#include <cmath>
#include <cstring>

namespace harness {
namespace layer_norm {

void layer_norm_golden(const float* __restrict__ x,
                       const float* __restrict__ gamma,
                       const float* __restrict__ beta,
                       float* __restrict__ y,
                       int64_t rows,
                       int64_t last_dim,
                       float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* x_row = x + r * last_dim;

        // Step 1: compute mean (float32 accumulation)
        double sum = 0.0;
        for (int64_t d = 0; d < last_dim; ++d) {
            sum += static_cast<double>(x_row[d]);
        }
        float mean = static_cast<float>(sum / static_cast<double>(last_dim));

        // Step 2: compute variance (float32 accumulation)
        double var_sum = 0.0;
        for (int64_t d = 0; d < last_dim; ++d) {
            float diff = x_row[d] - mean;
            var_sum += static_cast<double>(diff * diff);
        }
        float var = static_cast<float>(var_sum / static_cast<double>(last_dim));

        // Step 3: normalize and scale
        float inv_std = 1.0f / std::sqrt(var + eps);
        float* y_row = y + r * last_dim;
        for (int64_t d = 0; d < last_dim; ++d) {
            float norm = (x_row[d] - mean) * inv_std;
            y_row[d] = norm * gamma[d] + beta[d];
        }
    }
}

}  // namespace layer_norm
}  // namespace harness
