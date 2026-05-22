#include "axpy_scalar_affine_golden.hpp"
#include <stdexcept>

namespace harness {
namespace golden {
namespace axpy_scalar_affine {

void compute(const float* x, float* y, size_t num_elements,
             float alpha, float beta) {
    if (x == nullptr || y == nullptr) {
        throw std::invalid_argument("axpy_scalar_affine golden: null pointer");
    }
    for (size_t i = 0; i < num_elements; ++i) {
        y[i] = alpha * x[i] + beta;
    }
}

}  // namespace axpy_scalar_affine
}  // namespace golden
}  // namespace harness
