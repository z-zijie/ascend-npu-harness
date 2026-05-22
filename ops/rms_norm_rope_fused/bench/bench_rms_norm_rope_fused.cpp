#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "rms_norm_rope_fused_runner.hpp"
#include "harness/test_utils.hpp"

int main(int argc, char* argv[]) {
    int64_t B = 2, S = 128, H = 768, rotary_dim = 64;
    if (argc >= 5) { B = std::atoll(argv[1]); S = std::atoll(argv[2]); H = std::atoll(argv[3]); rotary_dim = std::atoll(argv[4]); }
    float eps = 1e-5f;

    auto x = harness::test::generate_random_float(B * S * H, 42, -2.0f, 2.0f);
    auto weight = harness::test::generate_random_float(H, 43, 0.5f, 1.5f);
    std::vector<float> cos, sin;
    harness::test::generate_rope_cos_sin(S, rotary_dim / 2, cos, sin);
    std::vector<float> out(B * S * H, 0.0f);

    const int warmup = 5;
    const int iters = 50;

    for (int i = 0; i < warmup; ++i) {
        harness::rms_norm_rope_fused::run_rms_norm_rope_fused_cpu(
            x.data(), weight.data(), cos.data(), sin.data(), out.data(),
            B, S, H, rotary_dim, eps);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        harness::rms_norm_rope_fused::run_rms_norm_rope_fused_cpu(
            x.data(), weight.data(), cos.data(), sin.data(), out.data(),
            B, S, H, rotary_dim, eps);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / iters;
    std::cout << "rms_norm_rope_fused [" << B << "," << S << "," << H
              << "] rotary=" << rotary_dim << " avg: " << avg_us << " us" << std::endl;
    return 0;
}
