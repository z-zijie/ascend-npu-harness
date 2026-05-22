#pragma once
#include "harness/shape.hpp"
#include "activation_gelu_silu_golden.hpp"

namespace harness {
namespace runner {
namespace activation_gelu_silu {

void run(const float* x, float* y, const Shape& shape, golden::activation_gelu_silu::ActivationType act);

}  // namespace activation_gelu_silu
}  // namespace runner
}  // namespace harness
