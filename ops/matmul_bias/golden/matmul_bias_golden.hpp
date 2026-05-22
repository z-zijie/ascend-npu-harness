#pragma once
#include <cstdint>

namespace harness {
namespace matmul_bias {

// C = A @ B + bias (optional)
// A = [M, K], B = [K, N], bias = [N] (optional, can be nullptr)
// C = [M, N]
//
// Uses float32 accumulation in the inner loop for numerical stability.
void matmul_bias_golden(const float* __restrict__ A,
                        const float* __restrict__ B,
                        const float* __restrict__ bias,  // nullptr or [N]
                        float* __restrict__ C,
                        int64_t M, int64_t K, int64_t N);

}  // namespace matmul_bias
}  // namespace harness
