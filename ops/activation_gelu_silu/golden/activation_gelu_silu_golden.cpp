#include "activation_gelu_silu_golden.hpp"
#include <cmath>
#include <stdexcept>

namespace harness {
namespace golden {
namespace activation_gelu_silu {

namespace {
    // sqrt(2/pi) constant for GELU tanh approximation
    constexpr float kSqrt2OverPi = 0.7978845608f;
    // Coefficient for x^3 term in GELU
    constexpr float kGeluCoeff = 0.044715f;
}

void compute(const float* x, float* y, size_t num_elements, ActivationType act) {
    if (x == nullptr || y == nullptr) {
        throw std::invalid_argument("activation_gelu_silu golden: null pointer");
    }

    switch (act) {
    case ActivationType::GELU:
        for (size_t i = 0; i < num_elements; ++i) {
            float val = x[i];
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float x3 = val * val * val;
            float inner = kSqrt2OverPi * (val + kGeluCoeff * x3);
            float tanh_val = std::tanh(inner);
            y[i] = 0.5f * val * (1.0f + tanh_val);
        }
        break;

    case ActivationType::SiLU:
        for (size_t i = 0; i < num_elements; ++i) {
            float val = x[i];
            // SiLU(x) = x / (1 + exp(-x))
            // Use numerically stable approach for large |x|
            if (val >= 0.0f) {
                float exp_neg = std::exp(-val);
                y[i] = val / (1.0f + exp_neg);
            } else {
                // x < 0: SiLU(x) = x * exp(x) / (1 + exp(x)) to avoid overflow
                float exp_val = std::exp(val);
                y[i] = val * exp_val / (1.0f + exp_val);
            }
        }
        break;
    }
}

}  // namespace activation_gelu_silu
}  // namespace golden
}  // namespace harness
