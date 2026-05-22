// Benchmark for elementwise_add
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "elementwise_add_golden.hpp"
#include "elementwise_add_runner.hpp"

int main(int argc, char** argv) {
    constexpr int warmup = 10;
    constexpr int iterations = 100;
    constexpr uint64_t seed = 20260522;

    // Default benchmark shape: [2, 3, 32, 128]
    harness::Shape shape({2, 3, 32, 128});
    size_t n = shape.num_elements(); // 24576

    if (argc >= 2) {
        // Parse shape from args: bench_elementwise_add D0 D1 D2 D3
        std::vector<int64_t> dims;
        for (int i = 1; i < argc; ++i) {
            dims.push_back(static_cast<int64_t>(std::atoll(argv[i])));
        }
        if (!dims.empty()) {
            shape = harness::Shape(dims);
            n = shape.num_elements();
        }
    }

    auto x = harness::test::generate_random_float(n, seed);
    auto y = harness::test::generate_random_float(n, seed + 1);
    std::vector<float> z(n, 0.0f);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        harness::runner::elementwise_add::run(x.data(), y.data(), z.data(), shape);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        harness::runner::elementwise_add::run(x.data(), y.data(), z.data(), shape);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / iterations;
    double bytes_processed = n * 3 * sizeof(float); // 2 reads + 1 write
    double gb_per_sec = (bytes_processed / (avg_us * 1e-6)) / 1e9;

    std::cout << "=== elementwise_add benchmark ===" << std::endl;
    std::cout << "  Shape: " << shape.to_string() << std::endl;
    std::cout << "  Elements: " << n << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg time: " << avg_us << " us" << std::endl;
    std::cout << "  Throughput: " << gb_per_sec << " GB/s" << std::endl;

    return 0;
}
