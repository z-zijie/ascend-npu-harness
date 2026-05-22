#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "flash_attention_fwd_fused_runner.hpp"
#include "harness/test_utils.hpp"

using namespace harness::flash_attention_fwd_fused;

int main(int argc, char* argv[]) {
    int64_t B = 1, H = 4, S_q = 64, S_k = 64, D = 64;
    if (argc >= 6) {
        B = std::atoll(argv[1]); H = std::atoll(argv[2]);
        S_q = std::atoll(argv[3]); S_k = std::atoll(argv[4]);
        D = std::atoll(argv[5]);
    }
    int64_t D_v = D;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    AttnMaskType mask = AttnMaskType::None;

    auto Q = harness::test::generate_random_float(B * H * S_q * D, 42, -1.0f, 1.0f);
    auto K = harness::test::generate_random_float(B * H * S_k * D, 43, -1.0f, 1.0f);
    auto V = harness::test::generate_random_float(B * H * S_k * D_v, 44, -1.0f, 1.0f);
    std::vector<float> O(B * H * S_q * D_v, 0.0f);

    const int warmup = 2;
    const int iters = 10;
    for (int i = 0; i < warmup; ++i) {
        run_flash_attention_fwd_fused_cpu(
            Q.data(), K.data(), V.data(), O.data(),
            B, H, S_q, S_k, D, D_v, scale, mask);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        run_flash_attention_fwd_fused_cpu(
            Q.data(), K.data(), V.data(), O.data(),
            B, H, S_q, S_k, D, D_v, scale, mask);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / (iters * 1000.0);
    std::cout << "flash_attn [" << B << "," << H << "," << S_q << "," << S_k << "," << D
              << "] avg: " << (avg_ms * 1000) << " us (" << avg_ms << " ms)" << std::endl;
    return 0;
}
