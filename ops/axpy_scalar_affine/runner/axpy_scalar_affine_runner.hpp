#pragma once
#include <cstddef>
#include "harness/shape.hpp"

namespace harness {
namespace runner {
namespace axpy_scalar_affine {

// Runner: y[i] = alpha * x[i] + beta
void run(const float* x, float* y, const Shape& shape, float alpha, float beta);

}  // namespace axpy_scalar_affine
}  // namespace runner
}  // namespace harness
