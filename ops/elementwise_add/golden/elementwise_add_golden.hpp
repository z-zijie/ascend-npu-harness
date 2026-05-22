#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace harness {
namespace golden {
namespace elementwise_add {

// CPU golden: z[i] = x[i] + y[i]
// All inputs must have the same number of elements.
// Always operates in float32 internally.

void compute(const float* x, const float* y, float* z, size_t num_elements);

// Convenience overload for vector inputs
inline void compute(const std::vector<float>& x,
                    const std::vector<float>& y,
                    std::vector<float>& z) {
    if (z.size() != x.size()) z.resize(x.size());
    compute(x.data(), y.data(), z.data(), x.size());
}

}  // namespace elementwise_add
}  // namespace golden
}  // namespace harness
