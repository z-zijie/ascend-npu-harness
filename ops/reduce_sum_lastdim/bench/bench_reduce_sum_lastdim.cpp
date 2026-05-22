// Benchmark for reduce_sum_lastdim
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "reduce_sum_lastdim_golden.hpp"
#include "reduce_sum_lastdim_runner.hpp"

int main(int argc, char** argv) {
    constexpr int warmup = 10;
    constexpr int iterations = 100;
    constexpr uint64_t seed = 20260522;

    // Default: 2D case [64, 4096]
    int64_t B = 1, M = 64, K = 4096;

    if (argc >= 3) {
        M = std::atoll(argv[1]);
        K = std::atoll(argv[2]);
        if (argc >= 4) {
            B = std::atoll(argv[3]);
        }
    }

    size_t n_in = static_cast<size_t>(B * M * K);
    size_t n_out = static_cast<size_t>(B * M);

    auto x = harness::test::generate_random_float(n_in, seed);
    std::vector<float> out(n_out, 0.0f);

    for (int i = 0; i < warmup; ++i) {
        harness::runner::reduce_sum_lastdim::run(x.data(), out.data(), B, M, K);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        harness::runner::reduce_sum_lastdim::run(x.data(), out.data(), B, M, K);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / iterations;
    double bytes_read = n_in * sizeof(float);
    double gb_per_sec = (bytes_read / (avg_us * 1e-6)) / 1e9;

    std::cout << "=== reduce_sum_lastdim benchmark ===" << std::endl;
    std::cout << "  Dims: B=" << B << " M=" << M << " K=" << K << std::endl;
    std::cout << "  Input elements: " << n_in << std::endl;
    std::cout << "  Output elements: " << n_out << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Avg time: " << avg_us << " us" << std::endl;
    std::cout << "  Read throughput: " << gb_per_sec << " GB/s" << std::endl;

    return 0;
}
