#pragma once
#include <cstdint>
#include "harness/shape.hpp"

namespace harness {
namespace runner {
namespace broadcast_binary {

// Runner: z = x * y + bias with broadcasting
// x: [B, C, H, W]
// y: [y_batch, C, 1, 1] where y_batch is 1 or B
// bias: [C]
void run(const float* x, const float* y, const float* bias, float* z,
         int64_t B, int64_t C, int64_t H, int64_t W, int64_t y_batch);

}  // namespace broadcast_binary
}  // namespace runner
}  // namespace harness
