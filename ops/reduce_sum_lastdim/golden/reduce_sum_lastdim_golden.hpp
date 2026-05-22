#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace harness {
namespace golden {
namespace reduce_sum_lastdim {

// CPU golden: sum over last dimension
//
// 2D case: out[i] = sum_{j=0}^{K-1} x[i * K + j]
// 3D case: out[b * M + i] = sum_{j=0}^{K-1} x[(b * M + i) * K + j]
//
// Accumulation is always done in float32 for numerical stability.
//
// Parameters:
//   x: input tensor, contiguous (row-major for 2D, batch-row-major for 3D)
//   out: output tensor
//   num_batches: 1 for 2D case [M, K], B for 3D case [B, M, K]
//   M: number of rows (2nd to last dimension)
//   K: reduction dimension (last dimension)

void compute(const float* x, float* out,
             int64_t num_batches, int64_t M, int64_t K);

// Convenience for 2D case
inline void compute_2d(const float* x, float* out, int64_t M, int64_t K) {
    compute(x, out, 1, M, K);
}

// Convenience for 3D case
inline void compute_3d(const float* x, float* out, int64_t B, int64_t M, int64_t K) {
    compute(x, out, B, M, K);
}

}  // namespace reduce_sum_lastdim
}  // namespace golden
}  // namespace harness
