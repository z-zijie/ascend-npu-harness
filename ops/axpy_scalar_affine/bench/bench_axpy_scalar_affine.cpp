// Benchmark for axpy_scalar_affine
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "axpy_scalar_affine_golden.hpp"
#include "axpy_scalar_affine_runner.hpp"

int main(int argc, char** argv) {
    constexpr int warmup = 10;
    constexpr int iterations = 100;
    constexpr uint64_t seed = 20260522;

    harness::Shape shape({32, 128});
    size_t n = shape.num_elements(); // 4096
    float alpha = 0.5f;
    float beta = -2.25f;

    if (argc >= 4) {
        alpha = std::atof(argv[1]);
        beta = std::atof(argv[2]);
        std::vector<int64_t> dims;
        for (int i = 3; i < argc; ++i) {
            dims.push_back(static_cast<int64_t>(std::atoll(argv[i])));
        }
        if (!dims.empty()) {
            shape = harness::Shape(dims);
            n = shape.num_elements();
        }
    }

    auto x = harness::test::generate_random_float(n, seed);
    std::vector<float> y(n, 0.0f);

    for (int i = 0; i < warmup; ++i) {
        harness::runner::axpy_scalar_affine::run(x.data(), y.data(), shape, alpha, beta);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        harness::runner::axpy_scalar_affine::run(x.data(), y.data(), shape, alpha, beta);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / iterations;
    double bytes_processed = n * 2 * sizeof(float);
    double gb_per_sec = (bytes_processed / (avg_us * 1e-6)) / 1e9;

    std::cout << "=== axpy_scalar_affine benchmark ===" << std::endl;
    std::cout << "  Shape: " << shape.to_string() << std::endl;
    std::cout << "  Elements: " << n << std::endl;
    std::cout << "  alpha=" << alpha << " beta=" << beta << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg time: " << avg_us << " us" << std::endl;
    std::cout << "  Throughput: " << gb_per_sec << " GB/s" << std::endl;

    return 0;
}
