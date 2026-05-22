#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"
#include "axpy_scalar_affine_golden.hpp"
#include "axpy_scalar_affine_runner.hpp"

namespace {

constexpr uint64_t kSeed = 20260522;
constexpr harness::accuracy::Tolerance kFp32Tol{1e-5, 1e-5};

// --- Golden correctness tests ---

TEST(AxpyScalarAffineGolden, Identity) {
    // alpha=1, beta=0 should copy
    std::vector<float> x = {1.0f, -2.5f, 3.14f, 0.0f};
    std::vector<float> y(4);
    harness::golden::axpy_scalar_affine::compute(x.data(), y.data(), 4, 1.0f, 0.0f);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(y[i], x[i]) << "at index " << i;
    }
}

TEST(AxpyScalarAffineGolden, ZeroOutput) {
    // alpha=0, beta=0 should zero output
    std::vector<float> x(100, 5.0f);
    std::vector<float> y(100);
    harness::golden::axpy_scalar_affine::compute(x.data(), y.data(), 100, 0.0f, 0.0f);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(y[i], 0.0f) << "at index " << i;
    }
}

TEST(AxpyScalarAffineGolden, ConstantOutput) {
    // alpha=0, beta=5 should produce all 5
    std::vector<float> x(10, 42.0f);
    std::vector<float> y(10);
    harness::golden::axpy_scalar_affine::compute(x.data(), y.data(), 10, 0.0f, 5.0f);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(y[i], 5.0f) << "at index " << i;
    }
}

TEST(AxpyScalarAffineGolden, SpecificValues) {
    std::vector<float> x = {2.0f, -1.0f, 0.0f, 3.0f};
    std::vector<float> y(4);
    // alpha=0.5, beta=-1.0 => y = 0.5*x - 1.0
    harness::golden::axpy_scalar_affine::compute(x.data(), y.data(), 4, 0.5f, -1.0f);
    EXPECT_FLOAT_EQ(y[0], 0.0f);
    EXPECT_FLOAT_EQ(y[1], -1.5f);
    EXPECT_FLOAT_EQ(y[2], -1.0f);
    EXPECT_FLOAT_EQ(y[3], 0.5f);
}

// --- Shape x Alpha/Beta coverage ---

struct AxpyTestCase {
    std::vector<int64_t> dims;
    float alpha;
    float beta;
};

class AxpyScalarAffineTest : public ::testing::TestWithParam<AxpyTestCase> {};

TEST_P(AxpyScalarAffineTest, RunnerMatchesGolden) {
    auto& tc = GetParam();
    harness::Shape shape(tc.dims);
    size_t n = shape.num_elements();

    auto x = harness::test::generate_random_float(n, kSeed);

    std::vector<float> y_golden(n);
    harness::golden::axpy_scalar_affine::compute(x.data(), y_golden.data(), n,
                                                  tc.alpha, tc.beta);

    std::vector<float> y_runner(n);
    harness::runner::axpy_scalar_affine::run(x.data(), y_runner.data(), shape,
                                              tc.alpha, tc.beta);

    auto report = harness::accuracy::CompareArrays(y_golden, y_runner, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString()
        << "\nshape=" << shape.to_string()
        << "\nalpha=" << tc.alpha << " beta=" << tc.beta
        << "\nseed=" << kSeed;
}

INSTANTIATE_TEST_SUITE_P(ShapesAndScalars, AxpyScalarAffineTest,
    ::testing::Values(
        // Test all shape x alpha/beta combinations from spec
        AxpyTestCase{{1}, 0.0f, 0.0f},
        AxpyTestCase{{1}, 1.0f, 1.0f},
        AxpyTestCase{{1}, -1.0f, -1.0f},
        AxpyTestCase{{1}, 0.5f, 0.5f},
        AxpyTestCase{{1}, -2.25f, -2.25f},
        AxpyTestCase{{17}, 0.0f, 1.0f},
        AxpyTestCase{{17}, 1.0f, 0.0f},
        AxpyTestCase{{17}, -1.0f, 0.5f},
        AxpyTestCase{{1024}, 0.5f, -1.0f},
        AxpyTestCase{{1024}, -2.25f, 1.0f},
        AxpyTestCase{{4097}, 0.5f, 0.0f},
        AxpyTestCase{{4097}, -2.25f, -2.25f},
        AxpyTestCase{{32, 128}, 0.0f, 0.0f},
        AxpyTestCase{{32, 128}, 1.0f, -1.0f},
        AxpyTestCase{{32, 128}, -2.25f, 0.5f}
    ));

// --- NPU skip test ---

TEST(AxpyScalarAffine, NpuSkipIfUnavailable) {
    if (!harness::IsAscendRuntimeAvailable() || !harness::HasUsableDevice()) {
        GTEST_SKIP() << "No NPU device available";
    }

    harness::Shape shape({1024});
    size_t n = shape.num_elements();
    auto x = harness::test::generate_random_float(n, kSeed);
    float alpha = 0.5f, beta = -1.0f;

    std::vector<float> y_golden(n);
    harness::golden::axpy_scalar_affine::compute(x.data(), y_golden.data(), n, alpha, beta);

    std::vector<float> y_npu(n);
    harness::runner::axpy_scalar_affine::run(x.data(), y_npu.data(), shape, alpha, beta);

    auto report = harness::accuracy::CompareArrays(y_golden, y_npu, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString() << "\nseed=" << kSeed;
}

}  // namespace
