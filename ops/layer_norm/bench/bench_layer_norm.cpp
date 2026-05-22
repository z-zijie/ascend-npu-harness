#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "layer_norm_runner.hpp"
#include "harness/test_utils.hpp"

int main(int argc, char* argv[]) {
    int64_t rows = 64;
    int64_t last_dim = 768;
    float eps = 1e-5f;
    if (argc >= 2) rows = std::atoll(argv[1]);
    if (argc >= 3) last_dim = std::atoll(argv[2]);

    size_t total_x = static_cast<size_t>(rows * last_dim);
    auto x = harness::test::generate_random_float(total_x, 42, -2.0f, 2.0f);
    auto gamma = harness::test::generate_random_float(last_dim, 43, 0.5f, 1.5f);
    auto beta = harness::test::generate_random_float(last_dim, 44, -0.5f, 0.5f);
    std::vector<float> y(total_x, 0.0f);

    const int warmup = 5;
    const int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < warmup; ++i) {
        harness::layer_norm::run_layer_norm_cpu(
            x.data(), gamma.data(), beta.data(), y.data(), rows, last_dim, eps);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        harness::layer_norm::run_layer_norm_cpu(
            x.data(), gamma.data(), beta.data(), y.data(), rows, last_dim, eps);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / iters;
    std::cout << "layer_norm [" << rows << "x" << last_dim << "] avg: " << avg_us << " us" << std::endl;
    return 0;
}
