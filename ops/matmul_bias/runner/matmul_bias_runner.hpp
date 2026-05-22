#pragma once
#include <cstdint>
#include "matmul_bias_golden.hpp"

namespace harness {
namespace matmul_bias {

// CPU runner — dispatches to golden
inline void run_matmul_bias_cpu(const float* A, const float* B,
                                 const float* bias,
                                 float* C,
                                 int64_t M, int64_t K, int64_t N) {
    matmul_bias_golden(A, B, bias, C, M, K, N);
}

}  // namespace matmul_bias
}  // namespace harness
