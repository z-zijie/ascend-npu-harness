#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"
#include "broadcast_binary_golden.hpp"
#include "broadcast_binary_runner.hpp"

namespace {

constexpr uint64_t kSeed = 20260522;
constexpr harness::accuracy::Tolerance kFp32Tol{1e-5, 1e-5};

// --- Golden correctness: hand-computed values ---

TEST(BroadcastBinaryGolden, Simple2x2) {
    // B=1, C=1, H=1, W=4
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {2.0f};       // y_batch=1, C=1
    std::vector<float> bias = {1.0f};    // C=1
    std::vector<float> z(4);

    harness::golden::broadcast_binary::compute(
        x.data(), y.data(), bias.data(), z.data(), 1, 1, 1, 4, 1);

    // z[i] = x[i] * y[0] + bias[0] = x[i] * 2 + 1
    EXPECT_FLOAT_EQ(z[0], 3.0f);
    EXPECT_FLOAT_EQ(z[1], 5.0f);
    EXPECT_FLOAT_EQ(z[2], 7.0f);
    EXPECT_FLOAT_EQ(z[3], 9.0f);
}

TEST(BroadcastBinaryGolden, MultiChannel) {
    // B=1, C=2, H=2, W=2
    std::vector<float> x = {
        1.f, 2.f, 3.f, 4.f,    // c=0
        5.f, 6.f, 7.f, 8.f     // c=1
    };
    std::vector<float> y = {10.f, 20.f};     // y_batch=1, C=2
    std::vector<float> bias = {0.5f, 1.0f};  // C=2
    std::vector<float> z(8);

    harness::golden::broadcast_binary::compute(
        x.data(), y.data(), bias.data(), z.data(), 1, 2, 2, 2, 1);

    // c=0: z = x * 10 + 0.5
    EXPECT_FLOAT_EQ(z[0], 10.5f);
    EXPECT_FLOAT_EQ(z[1], 20.5f);
    EXPECT_FLOAT_EQ(z[2], 30.5f);
    EXPECT_FLOAT_EQ(z[3], 40.5f);
    // c=1: z = x * 20 + 1.0
    EXPECT_FLOAT_EQ(z[4], 101.f);
    EXPECT_FLOAT_EQ(z[5], 121.f);
    EXPECT_FLOAT_EQ(z[6], 141.f);
    EXPECT_FLOAT_EQ(z[7], 161.f);
}

TEST(BroadcastBinaryGolden, PerBatchY) {
    // B=2, C=1, H=1, W=2, y_batch=2 (per-batch y)
    std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> y = {10.f, 100.f};    // y_batch=2, C=1
    std::vector<float> bias = {0.f};
    std::vector<float> z(4);

    harness::golden::broadcast_binary::compute(
        x.data(), y.data(), bias.data(), z.data(), 2, 1, 1, 2, 2);

    // b=0: y[0]=10 => z = x * 10
    EXPECT_FLOAT_EQ(z[0], 10.f);
    EXPECT_FLOAT_EQ(z[1], 20.f);
    // b=1: y[1]=100 => z = x * 100
    EXPECT_FLOAT_EQ(z[2], 300.f);
    EXPECT_FLOAT_EQ(z[3], 400.f);
}

// --- Shape coverage tests ---

struct BroadTestCase {
    int64_t B, C, H, W;
    int64_t y_batch; // 1 or B
};

class BroadcastBinaryShapeTest : public ::testing::TestWithParam<BroadTestCase> {};

TEST_P(BroadcastBinaryShapeTest, RunnerMatchesGolden) {
    auto& tc = GetParam();
    size_t n_x = static_cast<size_t>(tc.B * tc.C * tc.H * tc.W);
    size_t n_y = static_cast<size_t>(tc.y_batch * tc.C);
    size_t n_bias = static_cast<size_t>(tc.C);

    auto x = harness::test::generate_random_float(n_x, kSeed);
    auto y = harness::test::generate_random_float(n_y, kSeed + 1);
    auto bias = harness::test::generate_random_float(n_bias, kSeed + 2);

    std::vector<float> z_golden(n_x);
    harness::golden::broadcast_binary::compute(
        x.data(), y.data(), bias.data(), z_golden.data(),
        tc.B, tc.C, tc.H, tc.W, tc.y_batch);

    std::vector<float> z_runner(n_x);
    harness::runner::broadcast_binary::run(
        x.data(), y.data(), bias.data(), z_runner.data(),
        tc.B, tc.C, tc.H, tc.W, tc.y_batch);

    auto report = harness::accuracy::CompareArrays(z_golden, z_runner, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString()
        << "\nshape=[" << tc.B << "," << tc.C << "," << tc.H << "," << tc.W << "]"
        << " y_batch=" << tc.y_batch
        << "\nseed=" << kSeed;
}

INSTANTIATE_TEST_SUITE_P(AllShapes, BroadcastBinaryShapeTest,
    ::testing::Values(
        BroadTestCase{1, 1, 1, 1, 1},
        BroadTestCase{1, 3, 7, 7, 1},
        BroadTestCase{2, 16, 32, 32, 1},
        BroadTestCase{4, 64, 63, 63, 1},
        BroadTestCase{1, 3, 32, 64, 1},
        BroadTestCase{2, 1, 32, 32, 2},   // per-batch y
        BroadTestCase{1, 1, 63, 63, 1},   // non-power-of-2
        BroadTestCase{4, 16, 1, 1, 1},    // H=W=1 spatial
        BroadTestCase{1, 64, 7, 7, 1}     // large C, small spatial
    ));

// --- NPU skip test ---

TEST(BroadcastBinary, NpuSkipIfUnavailable) {
    if (!harness::IsAscendRuntimeAvailable() || !harness::HasUsableDevice()) {
        GTEST_SKIP() << "No NPU device available";
    }

    int64_t B = 2, C = 16, H = 32, W = 32, y_batch = 1;
    size_t n_x = static_cast<size_t>(B * C * H * W);
    auto x = harness::test::generate_random_float(n_x, kSeed);
    auto y = harness::test::generate_random_float(static_cast<size_t>(y_batch * C), kSeed + 1);
    auto bias = harness::test::generate_random_float(static_cast<size_t>(C), kSeed + 2);

    std::vector<float> z_golden(n_x);
    harness::golden::broadcast_binary::compute(
        x.data(), y.data(), bias.data(), z_golden.data(), B, C, H, W, y_batch);

    std::vector<float> z_npu(n_x);
    harness::runner::broadcast_binary::run(
        x.data(), y.data(), bias.data(), z_npu.data(), B, C, H, W, y_batch);

    auto report = harness::accuracy::CompareArrays(z_golden, z_npu, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString() << "\nseed=" << kSeed;
}

}  // namespace
