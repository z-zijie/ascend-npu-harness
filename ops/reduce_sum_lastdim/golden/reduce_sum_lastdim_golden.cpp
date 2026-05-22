#include "reduce_sum_lastdim_golden.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace harness {
namespace golden {
namespace reduce_sum_lastdim {

void compute(const float* x, float* out,
             int64_t num_batches, int64_t M, int64_t K) {
    if (x == nullptr || out == nullptr) {
        throw std::invalid_argument("reduce_sum_lastdim golden: null pointer");
    }
    if (num_batches <= 0 || M <= 0 || K <= 0) {
        throw std::invalid_argument("reduce_sum_lastdim golden: invalid dimensions");
    }

    // Total number of output elements: num_batches * M
    // Each output element reduces over K input elements
    // Accumulation in float32 (already the input dtype)
    //
    // For large K, use pairwise summation for better numerical stability.
    // Pairwise summation: recursively split the array and sum halves.

    for (int64_t b = 0; b < num_batches; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            // Base offset into input: (b * M + m) * K
            const float* row = x + ((b * M + m) * K);

            // Kahan compensated summation for numerical stability
            // This is important for large K values (>= 1024)
            float sum = 0.0f;
            float compensation = 0.0f;

            for (int64_t k = 0; k < K; ++k) {
                float y = row[k] - compensation;
                float t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }

            out[b * M + m] = sum;
        }
    }
}

}  // namespace reduce_sum_lastdim
}  // namespace golden
}  // namespace harness
