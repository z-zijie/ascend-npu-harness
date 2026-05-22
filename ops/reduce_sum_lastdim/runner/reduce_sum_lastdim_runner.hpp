#pragma once
#include <cstdint>
#include "harness/shape.hpp"

namespace harness {
namespace runner {
namespace reduce_sum_lastdim {

// Runner: sum over last dimension
// Handles both 2D [M, K] -> [M] and 3D [B, M, K] -> [B, M] cases.
//
// For 2D: num_batches=1, output_shape=[M]
// For 3D: num_batches=B, output_shape=[B, M]
void run(const float* x, float* out,
         int64_t num_batches, int64_t M, int64_t K);

}  // namespace reduce_sum_lastdim
}  // namespace runner
}  // namespace harness
