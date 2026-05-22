#include "matmul_bias_golden.hpp"

namespace harness {
namespace matmul_bias {

void matmul_bias_golden(const float* __restrict__ A,
                        const float* __restrict__ B,
                        const float* __restrict__ bias,
                        float* __restrict__ C,
                        int64_t M, int64_t K, int64_t N) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            // Accumulate in float32 for stability
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<double>(A[m * K + k]) *
                       static_cast<double>(B[k * N + n]);
            }
            int64_t idx = m * N + n;
            float val = static_cast<float>(acc);
            if (bias != nullptr) {
                val += bias[n];
            }
            C[idx] = val;
        }
    }
}

}  // namespace matmul_bias
}  // namespace harness
