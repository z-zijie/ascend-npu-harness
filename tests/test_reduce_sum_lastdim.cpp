#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include <numeric>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"
#include "reduce_sum_lastdim_golden.hpp"
#include "reduce_sum_lastdim_runner.hpp"

namespace {

constexpr uint64_t kSeed = 20260522;
constexpr harness::accuracy::Tolerance kFp32Tol{1e-4, 1e-4};

// --- Golden correctness: hand-computed values ---

TEST(ReduceSumLastdimGolden, Simple2D) {
    // M=2, K=3
    // x = [[1, 2, 3], [4, 5, 6]]
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2);

    harness::golden::reduce_sum_lastdim::compute_2d(x.data(), out.data(), 2, 3);

    EXPECT_FLOAT_EQ(out[0], 6.0f);   // 1+2+3
    EXPECT_FLOAT_EQ(out[1], 15.0f);  // 4+5+6
}

TEST(ReduceSumLastdimGolden, Simple3D) {
    // B=1, M=2, K=3
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2);

    harness::golden::reduce_sum_lastdim::compute_3d(x.data(), out.data(), 1, 2, 3);

    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 15.0f);
}

TEST(ReduceSumLastdimGolden, MultiBatch3D) {
    // B=2, M=1, K=4
    // x[0,:,:] = [1,2,3,4], x[1,:,:] = [5,6,7,8]
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out(2);

    harness::golden::reduce_sum_lastdim::compute_3d(x.data(), out.data(), 2, 1, 4);

    EXPECT_FLOAT_EQ(out[0], 10.0f);  // 1+2+3+4
    EXPECT_FLOAT_EQ(out[1], 26.0f);  // 5+6+7+8
}

TEST(ReduceSumLastdimGolden, SingleElementReduction) {
    // K=1: reduction over single element (identity)
    std::vector<float> x = {42.0f, -17.5f, 0.0f, 3.14f};
    std::vector<float> out(4);

    harness::golden::reduce_sum_lastdim::compute_2d(x.data(), out.data(), 4, 1);

    EXPECT_FLOAT_EQ(out[0], 42.0f);
    EXPECT_FLOAT_EQ(out[1], -17.5f);
    EXPECT_FLOAT_EQ(out[2], 0.0f);
    EXPECT_FLOAT_EQ(out[3], 3.14f);
}

TEST(ReduceSumLastdimGolden, Zeros) {
    std::vector<float> x(1024, 0.0f);
    std::vector<float> out(1);

    harness::golden::reduce_sum_lastdim::compute_2d(x.data(), out.data(), 1, 1024);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(ReduceSumLastdimGolden, AllOnesLargeK) {
    // Large K to exercise Kahan summation
    constexpr int64_t M = 1;
    constexpr int64_t K = 4096;
    std::vector<float> x(static_cast<size_t>(K), 1.0f);
    std::vector<float> out(static_cast<size_t>(M));

    harness::golden::reduce_sum_lastdim::compute_2d(x.data(), out.data(), M, K);
    EXPECT_NEAR(out[0], 4096.0f, 1e-2f);
}

// --- Shape coverage: 2D cases ---

struct Reduce2DTestCase {
    int64_t M, K;
};

class ReduceSumLastdim2DTest : public ::testing::TestWithParam<Reduce2DTestCase> {};

TEST_P(ReduceSumLastdim2DTest, RunnerMatchesGolden) {
    auto& tc = GetParam();
    size_t n_in = static_cast<size_t>(tc.M * tc.K);

    auto x = harness::test::generate_random_float(n_in, kSeed);

    std::vector<float> out_golden(static_cast<size_t>(tc.M));
    harness::golden::reduce_sum_lastdim::compute_2d(x.data(), out_golden.data(), tc.M, tc.K);

    std::vector<float> out_runner(static_cast<size_t>(tc.M));
    harness::runner::reduce_sum_lastdim::run(x.data(), out_runner.data(), 1, tc.M, tc.K);

    auto report = harness::accuracy::CompareArrays(out_golden, out_runner, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString()
        << "\nM=" << tc.M << " K=" << tc.K
        << "\nseed=" << kSeed;
}

INSTANTIATE_TEST_SUITE_P(Shape2D, ReduceSumLastdim2DTest,
    ::testing::Values(
        Reduce2DTestCase{1, 1},
        Reduce2DTestCase{1, 7},
        Reduce2DTestCase{1, 32},
        Reduce2DTestCase{1, 128},
        Reduce2DTestCase{1, 1024},
        Reduce2DTestCase{1, 4096},
        Reduce2DTestCase{3, 128},
        Reduce2DTestCase{64, 32},
        Reduce2DTestCase{64, 4096},
        Reduce2DTestCase{257, 7}
    ));

// --- Shape coverage: 3D cases ---

struct Reduce3DTestCase {
    int64_t B, M, K;
};

class ReduceSumLastdim3DTest : public ::testing::TestWithParam<Reduce3DTestCase> {};

TEST_P(ReduceSumLastdim3DTest, RunnerMatchesGolden) {
    auto& tc = GetParam();
    size_t n_in = static_cast<size_t>(tc.B * tc.M * tc.K);

    auto x = harness::test::generate_random_float(n_in, kSeed);

    std::vector<float> out_golden(static_cast<size_t>(tc.B * tc.M));
    harness::golden::reduce_sum_lastdim::compute_3d(x.data(), out_golden.data(), tc.B, tc.M, tc.K);

    std::vector<float> out_runner(static_cast<size_t>(tc.B * tc.M));
    harness::runner::reduce_sum_lastdim::run(x.data(), out_runner.data(), tc.B, tc.M, tc.K);

    auto report = harness::accuracy::CompareArrays(out_golden, out_runner, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString()
        << "\nB=" << tc.B << " M=" << tc.M << " K=" << tc.K
        << "\nseed=" << kSeed;
}

INSTANTIATE_TEST_SUITE_P(Shape3D, ReduceSumLastdim3DTest,
    ::testing::Values(
        Reduce3DTestCase{1, 1, 32},
        Reduce3DTestCase{2, 3, 128},
        Reduce3DTestCase{8, 64, 1024},
        Reduce3DTestCase{1, 64, 1},
        Reduce3DTestCase{2, 1, 4096},
        Reduce3DTestCase{8, 1, 7}
    ));

// --- Edge cases ---

TEST(ReduceSumLastdim, SingleElement2D) {
    // M=1, K=1: both dimensions are 1
    float x = 42.0f;
    float out = 0.0f;
    harness::golden::reduce_sum_lastdim::compute_2d(&x, &out, 1, 1);
    EXPECT_FLOAT_EQ(out, 42.0f);
}

TEST(ReduceSumLastdim, LargeKReductionStability) {
    // Test numerical stability for very large K
    constexpr int64_t M = 1;
    constexpr int64_t K = 100000;
    std::vector<float> x(static_cast<size_t>(K), 1e-6f);
    std::vector<float> out(static_cast<size_t>(M));

    harness::golden::reduce_sum_lastdim::compute_2d(x.data(), out.data(), M, K);

    // Expected: K * 1e-6 = 0.1
    // With Kahan summation, should be accurate
    EXPECT_NEAR(out[0], 0.1f, 1e-4f);
}

// --- NPU skip test ---

TEST(ReduceSumLastdim, NpuSkipIfUnavailable) {
    if (!harness::IsAscendRuntimeAvailable() || !harness::HasUsableDevice()) {
        GTEST_SKIP() << "No NPU device available";
    }

    int64_t B = 2, M = 64, K = 1024;
    size_t n_in = static_cast<size_t>(B * M * K);
    auto x = harness::test::generate_random_float(n_in, kSeed);

    std::vector<float> out_golden(static_cast<size_t>(B * M));
    harness::golden::reduce_sum_lastdim::compute_3d(x.data(), out_golden.data(), B, M, K);

    std::vector<float> out_npu(static_cast<size_t>(B * M));
    harness::runner::reduce_sum_lastdim::run(x.data(), out_npu.data(), B, M, K);

    auto report = harness::accuracy::CompareArrays(out_golden, out_npu, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString() << "\nseed=" << kSeed;
}

}  // namespace
