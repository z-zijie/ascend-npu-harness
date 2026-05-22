#include <iostream>
#include <chrono>
#include <vector>
#include "activation_gelu_silu_golden.hpp"
#include "activation_gelu_silu_runner.hpp"
#include "harness/test_utils.hpp"
#include "harness/shape.hpp"

using harness::golden::activation_gelu_silu::ActivationType;

int main() {
    std::cout << "Benchmark: activation_gelu_silu" << std::endl;
    harness::Shape shape({1024});
    size_t n = shape.num_elements();
    auto x = harness::test::generate_random_float(n);
    std::vector<float> y(n);

    // Warmup GELU
    for (int i = 0; i < 10; ++i)
        harness::runner::activation_gelu_silu::run(x.data(), y.data(), shape, ActivationType::GELU);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
        harness::runner::activation_gelu_silu::run(x.data(), y.data(), shape, ActivationType::GELU);
    auto t2 = std::chrono::high_resolution_clock::now();
    double gelu_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / 100.0;
    std::cout << "  GELU avg=" << gelu_us << " us" << std::endl;

    // Warmup SiLU
    for (int i = 0; i < 10; ++i)
        harness::runner::activation_gelu_silu::run(x.data(), y.data(), shape, ActivationType::SiLU);

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
        harness::runner::activation_gelu_silu::run(x.data(), y.data(), shape, ActivationType::SiLU);
    auto t4 = std::chrono::high_resolution_clock::now();
    double silu_us = std::chrono::duration<double, std::micro>(t4 - t3).count() / 100.0;
    std::cout << "  SiLU avg=" << silu_us << " us" << std::endl;
    std::cout << "  shape=" << shape.to_string() << std::endl;
    return 0;
}
