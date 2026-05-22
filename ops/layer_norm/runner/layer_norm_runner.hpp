#pragma once
#include <cstdint>
#include "layer_norm_golden.hpp"

namespace harness {
namespace layer_norm {

// CPU runner dispatches to golden
inline void run_layer_norm_cpu(const float* x, const float* gamma, const float* beta,
                                float* y, int64_t rows, int64_t last_dim, float eps) {
    layer_norm_golden(x, gamma, beta, y, rows, last_dim, eps);
}

}  // namespace layer_norm
}  // namespace harness
