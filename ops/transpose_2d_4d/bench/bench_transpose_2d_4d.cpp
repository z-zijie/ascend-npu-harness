#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "transpose_2d_4d_runner.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"

using namespace harness::transpose_2d_4d;
using harness::Shape;

int main(int argc, char* argv[]) {
    int64_t M = 128;
    int64_t N = 64;
    if (argc >= 3) {
        M = std::atoll(argv[1]);
        N = std::atoll(argv[2]);
    }

    size_t total = static_cast<size_t>(M * N);
    auto input = harness::test::generate_random_float(total, 42, -1.0f, 1.0f);
    std::vector<float> output(total, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    const int warmup = 5;
    const int iters = 20;
    for (int i = 0; i < warmup; ++i) {
        run_transpose_2d_cpu(input.data(), output.data(), M, N);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_transpose_2d_cpu(input.data(), output.data(), M, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / iters;

    std::cout << "transpose_2d [" << M << "x" << N << "] avg: " << avg_us << " us" << std::endl;
    return 0;
}
