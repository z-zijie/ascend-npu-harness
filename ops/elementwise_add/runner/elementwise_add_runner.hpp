#pragma once
#include <cstddef>
#include <cstdint>
#include "harness/shape.hpp"

namespace harness {
namespace runner {
namespace elementwise_add {

// Runner: dispatches to CPU golden or NPU kernel.
// In host-only mode, always uses golden.
// When NPU is available, launches the AscendC kernel.

void run(const float* x, const float* y, float* z, const Shape& shape);

}  // namespace elementwise_add
}  // namespace runner
}  // namespace harness
