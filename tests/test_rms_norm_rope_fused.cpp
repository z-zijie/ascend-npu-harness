#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"
#include "rms_norm_rope_fused_golden.hpp"
#include "rms_norm_rope_fused_runner.hpp"

namespace {

constexpr uint64_t kSeed = 20260522;

class RMSNormRoPETest : public ::testing::Test {
protected:
    void run_test(int64_t B, int64_t S, int64_t H, int64_t rotary_dim, float eps = 1e-5f) {
        size_t n = static_cast<size_t>(B * S * H);
        auto x = harness::test::generate_random_float(n, kSeed);
        auto weight = harness::test::generate_random_float(static_cast<size_t>(H), kSeed + 1);

        size_t rope_size = static_cast<size_t>(S * (rotary_dim / 2));
        std::vector<float> cos_vals(rope_size), sin_vals(rope_size);
        harness::test::generate_rope_cos_sin(static_cast<size_t>(S), static_cast<size_t>(rotary_dim / 2), cos_vals, sin_vals);

        std::vector<float> out_golden(n);
        harness::rms_norm_rope_fused::rms_norm_rope_fused_golden(
            x.data(), weight.data(), cos_vals.data(), sin_vals.data(),
            out_golden.data(), B, S, H, rotary_dim, eps);

        std::vector<float> out_runner(n);
        harness::rms_norm_rope_fused::run_rms_norm_rope_fused_cpu(
            x.data(), weight.data(), cos_vals.data(), sin_vals.data(),
            out_runner.data(), B, S, H, rotary_dim, eps);

        auto report = harness::accuracy::CompareArrays(out_golden, out_runner,
            harness::accuracy::Tolerance{1e-4, 1e-4});
        EXPECT_TRUE(report.passed) << report.ToString()
            << "\nB=" << B << " S=" << S << " H=" << H << " rotary=" << rotary_dim;
    }
};

TEST_F(RMSNormRoPETest, B1_S1_H64_R32)   { run_test(1, 1, 64, 32); }
TEST_F(RMSNormRoPETest, B1_S16_H128_R64)  { run_test(1, 16, 128, 64); }
TEST_F(RMSNormRoPETest, B2_S16_H256_R128) { run_test(2, 16, 256, 128); }
TEST_F(RMSNormRoPETest, B4_S128_H128_R64) { run_test(4, 128, 128, 64); }
TEST_F(RMSNormRoPETest, Eps1e6)           { run_test(1, 16, 128, 64, 1e-6f); }
TEST_F(RMSNormRoPETest, RotaryDimEqH)     { run_test(1, 16, 64, 64); }

}  // namespace
