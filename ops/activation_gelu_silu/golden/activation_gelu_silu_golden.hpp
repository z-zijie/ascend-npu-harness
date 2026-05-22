#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

namespace harness {
namespace golden {
namespace activation_gelu_silu {

// Activation type enum
enum class ActivationType {
    GELU,
    SiLU
};

// Parse activation type from string
inline ActivationType parse_activation(const std::string& name) {
    if (name == "gelu") return ActivationType::GELU;
    if (name == "silu") return ActivationType::SiLU;
    throw std::invalid_argument("Unknown activation: " + name);
}

// CPU golden: y[i] = activation(x[i])
//
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)

void compute(const float* x, float* y, size_t num_elements, ActivationType act);

// Convenience
inline void compute(const std::vector<float>& x, std::vector<float>& y,
                    ActivationType act) {
    if (y.size() != x.size()) y.resize(x.size());
    compute(x.data(), y.data(), x.size(), act);
}

}  // namespace activation_gelu_silu
}  // namespace golden
}  // namespace harness
