#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include "harness/accuracy.hpp"
#include "harness/test_utils.hpp"
#include "harness/shape.hpp"
#include "harness/skip.hpp"
#include "layer_norm_runner.hpp"

using namespace harness::layer_norm;
using harness::accuracy::Tolerance;

class LayerNormTest : public ::testing::Test {
protected:
    void run_test(int64_t rows, int64_t last_dim, float eps,
                  uint64_t seed = 20260522) {
        size_t total_x = static_cast<size_t>(rows * last_dim);
        auto x = harness::test::generate_random_float(total_x, seed, -3.0f, 3.0f);
        auto gamma = harness::test::generate_random_float(
            static_cast<size_t>(last_dim), seed + 1, 0.5f, 1.5f);
        auto beta = harness::test::generate_random_float(
            static_cast<size_t>(last_dim), seed + 2, -0.5f, 0.5f);
        std::vector<float> y_total(total_x, 0.0f);
        std::vector<float> expected(total_x, 0.0f);

        layer_norm_golden(x.data(), gamma.data(), beta.data(), expected.data(),
                          rows, last_dim, eps);
        run_layer_norm_cpu(x.data(), gamma.data(), beta.data(), y_total.data(),
                          rows, last_dim, eps);

        auto report = harness::accuracy::CompareArrays(expected, y_total,
            Tolerance{1e-4, 1e-4});
        EXPECT_TRUE(report.passed) << "layer_norm [" << rows << "," << last_dim
            << "] eps=" << eps << " failed: " << report.ToString();
    }
};

// --- Basic dimension tests ---
TEST_F(LayerNormTest, Dim16_M1)    { run_test(1, 16, 1e-5f); }
TEST_F(LayerNormTest, Dim64_M3)    { run_test(3, 64, 1e-5f); }
TEST_F(LayerNormTest, Dim128_M1)   { run_test(1, 128, 1e-5f); }
TEST_F(LayerNormTest, Dim768_M64)  { run_test(64, 768, 1e-5f); }
TEST_F(LayerNormTest, Dim1024_M4)  { run_test(4, 1024, 1e-5f); }
TEST_F(LayerNormTest, Dim4096_M1)  { run_test(1, 4096, 1e-5f); }
TEST_F(LayerNormTest, Dim64_M257)  { run_test(257, 64, 1e-5f); }

// --- 3D-like shapes (B,S as rows) ---
TEST_F(LayerNormTest, B1S1_H64)  { run_test(1, 64, 1e-5f); }    // B=1,S=1 -> rows=1
TEST_F(LayerNormTest, B1S16_H768) { run_test(16, 768, 1e-5f); } // B=1,S=16 -> rows=16
TEST_F(LayerNormTest, B2S16_H256) { run_test(32, 256, 1e-5f); } // B=2,S=16 -> rows=32
TEST_F(LayerNormTest, B4S128_H128) { run_test(512, 128, 1e-5f); } // B=4,S=128 -> rows=512

// --- Eps values ---
TEST_F(LayerNormTest, Eps1e5_64x768)  { run_test(64, 768, 1e-5f); }
TEST_F(LayerNormTest, Eps1e6_64x768)  { run_test(64, 768, 1e-6f); }

// --- Special cases ---

// Constant row: all elements equal - mean should equal the constant, var=0
TEST_F(LayerNormTest, ConstantRow) {
    int64_t rows = 4;
    int64_t dim = 128;
    float eps = 1e-5f;
    float const_val = 3.5f;

    std::vector<float> x(static_cast<size_t>(rows * dim), const_val);
    auto gamma = harness::test::generate_random_float(dim, 99, 0.5f, 1.5f);
    auto beta = harness::test::generate_random_float(dim, 100, -0.5f, 0.5f);
    std::vector<float> y(static_cast<size_t>(rows * dim), 0.0f);
    std::vector<float> expected(static_cast<size_t>(rows * dim), 0.0f);

    layer_norm_golden(x.data(), gamma.data(), beta.data(), expected.data(),
                      rows, dim, eps);
    run_layer_norm_cpu(x.data(), gamma.data(), beta.data(), y.data(),
                      rows, dim, eps);

    auto report = harness::accuracy::CompareArrays(expected, y,
        Tolerance{1e-4, 1e-4});
    EXPECT_TRUE(report.passed) << "Constant row test failed: " << report.ToString();

    // For constant input, output should equal beta (since norm = 0)
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t d = 0; d < dim; ++d) {
            EXPECT_NEAR(y[r * dim + d], beta[d], 1e-4f)
                << "Constant row output mismatch at row=" << r << " dim=" << d;
        }
    }
}

// Low-variance row: values very close together
TEST_F(LayerNormTest, LowVarianceRow) {
    int64_t rows = 2;
    int64_t dim = 64;
    float eps = 1e-5f;

    std::vector<float> x(static_cast<size_t>(rows * dim));
    for (int64_t r = 0; r < rows; ++r) {
        float base = (r == 0) ? 5.0f : -3.0f;
        for (int64_t d = 0; d < dim; ++d) {
            x[static_cast<size_t>(r * dim + d)] = base + 1e-6f * static_cast<float>(d);
        }
    }
    auto gamma = harness::test::generate_random_float(dim, 101, 0.5f, 1.5f);
    auto beta = harness::test::generate_random_float(dim, 102, -0.5f, 0.5f);
    std::vector<float> y(static_cast<size_t>(rows * dim), 0.0f);
    std::vector<float> expected(static_cast<size_t>(rows * dim), 0.0f);

    layer_norm_golden(x.data(), gamma.data(), beta.data(), expected.data(),
                      rows, dim, eps);
    run_layer_norm_cpu(x.data(), gamma.data(), beta.data(), y.data(),
                      rows, dim, eps);

    auto report = harness::accuracy::CompareArrays(expected, y,
        Tolerance{1e-4, 1e-4});
    EXPECT_TRUE(report.passed) << "Low variance test failed: " << report.ToString();
}
