#include "elementwise_add_golden.hpp"
#include <stdexcept>

namespace harness {
namespace golden {
namespace elementwise_add {

void compute(const float* x, const float* y, float* z, size_t num_elements) {
    if (x == nullptr || y == nullptr || z == nullptr) {
        throw std::invalid_argument("elementwise_add golden: null pointer");
    }
    for (size_t i = 0; i < num_elements; ++i) {
        z[i] = x[i] + y[i];
    }
}

}  // namespace elementwise_add
}  // namespace golden
}  // namespace harness
