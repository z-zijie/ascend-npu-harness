#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"
#include "elementwise_add_golden.hpp"
#include "elementwise_add_runner.hpp"

namespace {

constexpr uint64_t kSeed = 20260522;
constexpr harness::accuracy::Tolerance kFp32Tol{1e-5, 1e-5};

// --- Golden correctness tests ---

TEST(ElementwiseAddGolden, SimpleValues) {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> z(4);
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};

    harness::golden::elementwise_add::compute(x.data(), y.data(), z.data(), 4);
    auto report = harness::accuracy::CompareArrays(expected, z, kFp32Tol);
    EXPECT_TRUE(report.passed) << report.ToString() << "\nseed=" << kSeed;
}

TEST(ElementwiseAddGolden, Zeros) {
    std::vector<float> x(1024, 0.0f);
    std::vector<float> y(1024, 0.0f);
    std::vector<float> z(1024, -1.0f);

    harness::golden::elementwise_add::compute(x.data(), y.data(), z.data(), 1024);
    for (size_t i = 0; i < 1024; ++i) {
        EXPECT_FLOAT_EQ(z[i], 0.0f) << "at index " << i;
    }
}

TEST(ElementwiseAddGolden, Negatives) {
    std::vector<float> x = {-1.0f, -2.5f, 3.0f};
    std::vector<float> y = {1.0f, 2.5f, -3.0f};
    std::vector<float> z(3);

    harness::golden::elementwise_add::compute(x.data(), y.data(), z.data(), 3);
    EXPECT_FLOAT_EQ(z[0], 0.0f);
    EXPECT_FLOAT_EQ(z[1], 0.0f);
    EXPECT_FLOAT_EQ(z[2], 0.0f);
}

// --- Shape coverage tests ---

struct ShapeTestCase {
    std::vector<int64_t> dims;
};

class ElementwiseAddShapeTest : public ::testing::TestWithParam<ShapeTestCase> {};

TEST_P(ElementwiseAddShapeTest, RunnerMatchesGolden) {
    auto& tc = GetParam();
    harness::Shape shape(tc.dims);
    size_t n = shape.num_elements();

    auto x = harness::test::generate_random_float(n, kSeed);
    auto y = harness::test::generate_random_float(n, kSeed + 1);

    // Golden
    std::vector<float> z_golden(n);
    harness::golden::elementwise_add::compute(x.data(), y.data(), z_golden.data(), n);

    // Runner
    std::vector<float> z_runner(n);
    harness::runner::elementwise_add::run(x.data(), y.data(), z_runner.data(), shape);

    auto report = harness::accuracy::CompareArrays(z_golden, z_runner, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString() << "\nshape=" << shape.to_string() << "\nseed=" << kSeed;
}

INSTANTIATE_TEST_SUITE_P(AllShapes, ElementwiseAddShapeTest,
    ::testing::Values(
        ShapeTestCase{{1}},
        ShapeTestCase{{8}},
        ShapeTestCase{{1024}},
        ShapeTestCase{{3, 7}},
        ShapeTestCase{{4, 16, 64}},
        ShapeTestCase{{2, 3, 32, 128}}
    ));

// --- NPU test (skip if unavailable) ---

TEST(ElementwiseAdd, NpuSkipIfUnavailable) {
    if (!harness::IsAscendRuntimeAvailable() || !harness::HasUsableDevice()) {
        GTEST_SKIP() << "No NPU device available";
    }

    // If NPU is available, run a comparison test
    harness::Shape shape({2, 3, 32, 128});
    size_t n = shape.num_elements();
    auto x = harness::test::generate_random_float(n, kSeed);
    auto y = harness::test::generate_random_float(n, kSeed + 1);

    std::vector<float> z_golden(n);
    harness::golden::elementwise_add::compute(x.data(), y.data(), z_golden.data(), n);

    std::vector<float> z_npu(n);
    harness::runner::elementwise_add::run(x.data(), y.data(), z_npu.data(), shape);

    auto report = harness::accuracy::CompareArrays(z_golden, z_npu, kFp32Tol);
    EXPECT_TRUE(report.passed)
        << report.ToString() << "\nseed=" << kSeed;
}

// --- Edge case: single element ---

TEST(ElementwiseAdd, SingleElement) {
    float x = 3.14f, y = 2.86f, z = 0.0f;
    harness::golden::elementwise_add::compute(&x, &y, &z, 1);
    EXPECT_NEAR(z, 6.0f, 1e-5);
}

}  // namespace
