#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace harness {
namespace golden {
namespace axpy_scalar_affine {

// CPU golden: y[i] = alpha * x[i] + beta
// alpha and beta are scalar constants broadcast to all elements.
// Always operates in float32 internally.

void compute(const float* x, float* y, size_t num_elements,
             float alpha, float beta);

// Convenience overload for vector inputs
inline void compute(const std::vector<float>& x,
                    std::vector<float>& y,
                    float alpha, float beta) {
    if (y.size() != x.size()) y.resize(x.size());
    compute(x.data(), y.data(), x.size(), alpha, beta);
}

}  // namespace axpy_scalar_affine
}  // namespace golden
}  // namespace harness
