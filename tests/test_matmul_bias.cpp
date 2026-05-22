#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include "harness/accuracy.hpp"
#include "harness/test_utils.hpp"
#include "harness/shape.hpp"
#include "harness/skip.hpp"
#include "matmul_bias_runner.hpp"

using namespace harness::matmul_bias;
using harness::accuracy::Tolerance;

class MatmulBiasTest : public ::testing::Test {
protected:
    void run_test(int64_t M, int64_t K, int64_t N, bool with_bias,
                  uint64_t seed = 20260522) {
        size_t size_A = static_cast<size_t>(M * K);
        size_t size_B = static_cast<size_t>(K * N);
        size_t size_C = static_cast<size_t>(M * N);

        auto A = harness::test::generate_random_float(size_A, seed, -2.0f, 2.0f);
        auto B = harness::test::generate_random_float(size_B, seed + 1, -2.0f, 2.0f);
        auto bias_vec = harness::test::generate_random_float(
            static_cast<size_t>(N), seed + 2, -0.5f, 0.5f);
        std::vector<float> C(size_C, 0.0f);
        std::vector<float> expected(size_C, 0.0f);

        const float* bias_ptr = with_bias ? bias_vec.data() : nullptr;

        matmul_bias_golden(A.data(), B.data(), bias_ptr, expected.data(), M, K, N);
        run_matmul_bias_cpu(A.data(), B.data(), bias_ptr, C.data(), M, K, N);

        auto report = harness::accuracy::CompareArrays(expected, C,
            Tolerance{1e-4, 1e-4});
        EXPECT_TRUE(report.passed) << "matmul_bias [" << M << "," << K << ","
            << N << "] bias=" << (with_bias ? "yes" : "no") << " failed: "
            << report.ToString();
    }
};

// --- Without bias ---
TEST_F(MatmulBiasTest, NoBias_M1_N1_K1)     { run_test(1, 1, 1, false); }
TEST_F(MatmulBiasTest, NoBias_M16_N16_K16)  { run_test(16, 16, 16, false); }
TEST_F(MatmulBiasTest, NoBias_M64_N64_K64)  { run_test(64, 64, 64, false); }
TEST_F(MatmulBiasTest, NoBias_M128_N128_K128) { run_test(128, 128, 128, false); }
TEST_F(MatmulBiasTest, NoBias_M256_N16_K64)  { run_test(256, 64, 16, false); }
TEST_F(MatmulBiasTest, NoBias_M1_N64_K1024)  { run_test(1, 1024, 64, false); } // skinny GEMM

// --- With bias ---
TEST_F(MatmulBiasTest, WithBias_M1_N1_K1)     { run_test(1, 1, 1, true); }
TEST_F(MatmulBiasTest, WithBias_M16_N16_K16)  { run_test(16, 16, 16, true); }
TEST_F(MatmulBiasTest, WithBias_M64_N128_K32) { run_test(64, 128, 32, true); }
TEST_F(MatmulBiasTest, WithBias_M128_N128_K128) { run_test(128, 128, 128, true); }
TEST_F(MatmulBiasTest, WithBias_M256_N64_K256)  { run_test(256, 256, 64, true); }

// --- Square GEMM ---
TEST_F(MatmulBiasTest, Square_128x128x128_NoBias)  { run_test(128, 128, 128, false); }
TEST_F(MatmulBiasTest, Square_128x128x128_WithBias) { run_test(128, 128, 128, true); }

// --- Skinny GEMM with bias ---
TEST_F(MatmulBiasTest, Skinny_M1_K1024_N64)  { run_test(1, 1024, 64, true); }

// --- Edge cases ---
TEST_F(MatmulBiasTest, Edge_M64_N1_K64)  { run_test(64, 64, 1, false); }   // single column
TEST_F(MatmulBiasTest, Edge_M1_N64_K1)   { run_test(1, 1, 64, false); }    // single row, trivial K
TEST_F(MatmulBiasTest, Edge_M1_N1_K1024)  { run_test(1, 1024, 1, true); } // large K dot product

// --- Numerical verification: identity test ---
// If A = identity, B = identity (with padding), C should be identity (+ bias)
TEST_F(MatmulBiasTest, IdentityNoBias) {
    int64_t M = 4, K = 4, N = 4;
    std::vector<float> A(M * K, 0.0f);
    std::vector<float> B(K * N, 0.0f);
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> expected(M * N, 0.0f);

    // A: identity
    for (int64_t i = 0; i < std::min(M, K); ++i) A[i * K + i] = 1.0f;
    // B: identity
    for (int64_t i = 0; i < std::min(K, N); ++i) B[i * N + i] = 1.0f;

    matmul_bias_golden(A.data(), B.data(), nullptr, expected.data(), M, K, N);
    run_matmul_bias_cpu(A.data(), B.data(), nullptr, C.data(), M, K, N);

    auto report = harness::accuracy::CompareArrays(expected, C,
        Tolerance{1e-5, 1e-5});
    EXPECT_TRUE(report.passed) << report.ToString();

    // Expected: identity (product of two identity matrices)
    for (int64_t i = 0; i < std::min(M, N); ++i) {
        EXPECT_NEAR(C[i * N + i], 1.0f, 1e-5f) << "diagonal element " << i;
    }
}
