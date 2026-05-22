#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "matmul_bias_runner.hpp"
#include "harness/test_utils.hpp"

int main(int argc, char* argv[]) {
    int64_t M = 128, K = 128, N = 128;
    bool with_bias = true;
    if (argc >= 4) { M = std::atoll(argv[1]); K = std::atoll(argv[2]); N = std::atoll(argv[3]); }

    auto A = harness::test::generate_random_float(M * K, 42, -1.0f, 1.0f);
    auto B = harness::test::generate_random_float(K * N, 43, -1.0f, 1.0f);
    auto bias = harness::test::generate_random_float(N, 44, -0.5f, 0.5f);
    std::vector<float> C(M * N, 0.0f);

    const int warmup = 3;
    const int iters = 20;

    for (int i = 0; i < warmup; ++i) {
        harness::matmul_bias::run_matmul_bias_cpu(
            A.data(), B.data(), with_bias ? bias.data() : nullptr, C.data(), M, K, N);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        harness::matmul_bias::run_matmul_bias_cpu(
            A.data(), B.data(), with_bias ? bias.data() : nullptr, C.data(), M, K, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / iters;
    double gflops = (2.0 * M * K * N) / (avg_us * 1e3);
    std::cout << "matmul_bias [" << M << "x" << K << "x" << N
              << "] avg: " << avg_us << " us, " << gflops << " GFLOPS" << std::endl;
    return 0;
}
