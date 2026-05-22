// Benchmark for broadcast_binary
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "broadcast_binary_golden.hpp"
#include "broadcast_binary_runner.hpp"

int main(int argc, char** argv) {
    constexpr int warmup = 10;
    constexpr int iterations = 100;
    constexpr uint64_t seed = 20260522;

    int64_t B = 2, C = 16, H = 32, W = 32;
    int64_t y_batch = 1;

    if (argc >= 5) {
        B = std::atoll(argv[1]);
        C = std::atoll(argv[2]);
        H = std::atoll(argv[3]);
        W = std::atoll(argv[4]);
    }
    if (argc >= 6) y_batch = std::atoll(argv[5]);

    size_t n_x = static_cast<size_t>(B * C * H * W);
    auto x = harness::test::generate_random_float(n_x, seed);
    auto y = harness::test::generate_random_float(static_cast<size_t>(y_batch * C), seed + 1);
    auto bias = harness::test::generate_random_float(static_cast<size_t>(C), seed + 2);
    std::vector<float> z(n_x, 0.0f);

    for (int i = 0; i < warmup; ++i) {
        harness::runner::broadcast_binary::run(x.data(), y.data(), bias.data(),
            z.data(), B, C, H, W, y_batch);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        harness::runner::broadcast_binary::run(x.data(), y.data(), bias.data(),
            z.data(), B, C, H, W, y_batch);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / iterations;
    double bytes_processed = n_x * 3 * sizeof(float);
    double gb_per_sec = (bytes_processed / (avg_us * 1e-6)) / 1e9;

    harness::Shape sh({B, C, H, W});
    std::cout << "=== broadcast_binary benchmark ===" << std::endl;
    std::cout << "  Shape: " << sh.to_string() << std::endl;
    std::cout << "  y_batch=" << y_batch << std::endl;
    std::cout << "  Elements: " << n_x << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg time: " << avg_us << " us" << std::endl;
    std::cout << "  Throughput: " << gb_per_sec << " GB/s" << std::endl;

    return 0;
}
