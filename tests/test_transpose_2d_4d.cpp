#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include "harness/accuracy.hpp"
#include "harness/test_utils.hpp"
#include "harness/shape.hpp"
#include "harness/skip.hpp"
#include "transpose_2d_4d_runner.hpp"

using namespace harness::transpose_2d_4d;
using harness::accuracy::Tolerance;

class Transpose2DTest : public ::testing::Test {
protected:
    void run_float_test(int64_t M, int64_t N) {
        size_t total = static_cast<size_t>(M * N);
        auto input = harness::test::generate_random_float(total, 20260522, -10.0f, 10.0f);
        std::vector<float> output(total, 0.0f);
        std::vector<float> expected(total, 0.0f);

        transpose_2d_golden(input.data(), expected.data(), M, N);
        run_transpose_2d_cpu(input.data(), output.data(), M, N);

        auto report = harness::accuracy::CompareArrays(expected, output,
            Tolerance{1e-5, 1e-5});
        EXPECT_TRUE(report.passed) << "transpose_2d [" << M << "," << N
            << "] failed: " << report.ToString();
    }

    void run_int32_test(int64_t M, int64_t N) {
        size_t total = static_cast<size_t>(M * N);
        auto input = harness::test::generate_random_int(total, 20260522, -500, 500);
        std::vector<int32_t> output(total, 0);
        std::vector<int32_t> expected(total, 0);

        transpose_2d_golden(input.data(), expected.data(), M, N);
        run_transpose_2d_cpu(input.data(), output.data(), M, N);

        auto report = harness::accuracy::CompareArrays(expected, output,
            Tolerance{0.0, 0.0});  // exact match for int32
        EXPECT_TRUE(report.passed) << "transpose_2d int32 [" << M << "," << N
            << "] failed: " << report.ToString();
    }
};

// --- 2D transpose tests (float32) ---
TEST_F(Transpose2DTest, Float2D_1x1)  { run_float_test(1, 1); }
TEST_F(Transpose2DTest, Float2D_3x7)  { run_float_test(3, 7); }
TEST_F(Transpose2DTest, Float2D_32x32) { run_float_test(32, 32); }
TEST_F(Transpose2DTest, Float2D_128x64) { run_float_test(128, 64); }
TEST_F(Transpose2DTest, Float2D_257x129) { run_float_test(257, 129); }

// --- 2D transpose tests (int32, exact comparison) ---
TEST_F(Transpose2DTest, Int32_2D_1x1)  { run_int32_test(1, 1); }
TEST_F(Transpose2DTest, Int32_2D_3x7)  { run_int32_test(3, 7); }
TEST_F(Transpose2DTest, Int32_2D_32x32) { run_int32_test(32, 32); }
TEST_F(Transpose2DTest, Int32_2D_128x64) { run_int32_test(128, 64); }
TEST_F(Transpose2DTest, Int32_2D_257x129) { run_int32_test(257, 129); }

// --- 4D NCHW <-> NHWC tests ---

class Transpose4DTest : public ::testing::Test {
protected:
    void run_nchw_to_nhwc(int64_t N, int64_t C, int64_t H, int64_t W) {
        size_t total = static_cast<size_t>(N * C * H * W);
        auto input = harness::test::generate_random_float(total, 42, -10.0f, 10.0f);
        std::vector<float> output(total, 0.0f);
        std::vector<float> expected(total, 0.0f);

        transpose_4d_nchw_to_nhwc_golden(input.data(), expected.data(), N, C, H, W);
        run_transpose_4d_nchw_to_nhwc_cpu(input.data(), output.data(), N, C, H, W);

        auto report = harness::accuracy::CompareArrays(expected, output,
            Tolerance{1e-5, 1e-5});
        EXPECT_TRUE(report.passed) << "NCHW->NHWC [" << N << "," << C << ","
            << H << "," << W << "] failed: " << report.ToString();
    }

    void run_nhwc_to_nchw(int64_t N, int64_t C, int64_t H, int64_t W) {
        size_t total = static_cast<size_t>(N * C * H * W);
        auto input = harness::test::generate_random_float(total, 43, -10.0f, 10.0f);
        std::vector<float> output(total, 0.0f);
        std::vector<float> expected(total, 0.0f);

        transpose_4d_nhwc_to_nchw_golden(input.data(), expected.data(), N, C, H, W);
        run_transpose_4d_nhwc_to_nchw_cpu(input.data(), output.data(), N, C, H, W);

        auto report = harness::accuracy::CompareArrays(expected, output,
            Tolerance{1e-5, 1e-5});
        EXPECT_TRUE(report.passed) << "NHWC->NCHW [" << N << "," << C << ","
            << H << "," << W << "] failed: " << report.ToString();
    }

    void run_round_trip(int64_t N, int64_t C, int64_t H, int64_t W) {
        size_t total = static_cast<size_t>(N * C * H * W);
        auto input = harness::test::generate_random_float(total, 44, -10.0f, 10.0f);
        std::vector<float> tmp(total, 0.0f);
        std::vector<float> result(total, 0.0f);

        // NCHW -> NHWC -> NCHW should give original
        transpose_4d_nchw_to_nhwc_golden(input.data(), tmp.data(), N, C, H, W);
        transpose_4d_nhwc_to_nchw_golden(tmp.data(), result.data(), N, C, H, W);

        auto report = harness::accuracy::CompareArrays(input, result,
            Tolerance{1e-5, 1e-5});
        EXPECT_TRUE(report.passed) << "Round-trip (NCHW->NHWC->NCHW) [" << N
            << "," << C << "," << H << "," << W << "] failed: "
            << report.ToString();
    }
};

// --- 4D NCHW -> NHWC ---
TEST_F(Transpose4DTest, NCHW2NHWC_1x3x224x224) { run_nchw_to_nhwc(1, 3, 224, 224); }
TEST_F(Transpose4DTest, NCHW2NHWC_2x16x32x32)  { run_nchw_to_nhwc(2, 16, 32, 32); }
TEST_F(Transpose4DTest, NCHW2NHWC_4x7x13x17)   { run_nchw_to_nhwc(4, 7, 13, 17); }

// --- 4D NHWC -> NCHW ---
TEST_F(Transpose4DTest, NHWC2NCHW_1x3x224x224) { run_nhwc_to_nchw(1, 3, 224, 224); }
TEST_F(Transpose4DTest, NHWC2NCHW_2x16x32x32)  { run_nhwc_to_nchw(2, 16, 32, 32); }
TEST_F(Transpose4DTest, NHWC2NCHW_4x7x13x17)   { run_nhwc_to_nchw(4, 7, 13, 17); }

// --- Round-trip tests ---
TEST_F(Transpose4DTest, RoundTrip_1x3x224x224) { run_round_trip(1, 3, 224, 224); }
TEST_F(Transpose4DTest, RoundTrip_2x16x32x32)  { run_round_trip(2, 16, 32, 32); }
TEST_F(Transpose4DTest, RoundTrip_4x7x13x17)   { run_round_trip(4, 7, 13, 17); }

// --- Int32 4D tests ---
TEST(Transpose4DInt32, NCHW2NHWC_Int32) {
    int64_t N = 2, C = 3, H = 8, W = 8;
    size_t total = static_cast<size_t>(N * C * H * W);
    auto input = harness::test::generate_random_int(total, 77, -100, 100);
    std::vector<int32_t> expected(total, 0);
    std::vector<int32_t> output(total, 0);

    transpose_4d_nchw_to_nhwc_golden(input.data(), expected.data(), N, C, H, W);
    run_transpose_4d_nchw_to_nhwc_cpu(input.data(), output.data(), N, C, H, W);

    auto report = harness::accuracy::CompareArrays(expected, output, Tolerance{0.0, 0.0});
    EXPECT_TRUE(report.passed) << report.ToString();
}

TEST(Transpose4DInt32, NHWC2NCHW_Int32) {
    int64_t N = 2, C = 4, H = 5, W = 6;
    size_t total = static_cast<size_t>(N * C * H * W);
    auto input = harness::test::generate_random_int(total, 78, -100, 100);
    std::vector<int32_t> expected(total, 0);
    std::vector<int32_t> output(total, 0);

    transpose_4d_nhwc_to_nchw_golden(input.data(), expected.data(), N, C, H, W);
    run_transpose_4d_nhwc_to_nchw_cpu(input.data(), output.data(), N, C, H, W);

    auto report = harness::accuracy::CompareArrays(expected, output, Tolerance{0.0, 0.0});
    EXPECT_TRUE(report.passed) << report.ToString();
}
