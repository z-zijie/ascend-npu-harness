#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"
#include "activation_gelu_silu_golden.hpp"
#include "activation_gelu_silu_runner.hpp"

namespace {

using harness::golden::activation_gelu_silu::ActivationType;
constexpr uint64_t kSeed = 20260522;

TEST(ActivationGeluSiluGolden, GeluKnownValues) {
    float x = 0.0f, y;
    harness::golden::activation_gelu_silu::compute(&x, &y, 1, ActivationType::GELU);
    EXPECT_NEAR(y, 0.0f, 1e-5);
}

TEST(ActivationGeluSiluGolden, SiluKnownValues) {
    float x = 0.0f, y;
    harness::golden::activation_gelu_silu::compute(&x, &y, 1, ActivationType::SiLU);
    EXPECT_NEAR(y, 0.0f, 1e-5);
}

TEST(ActivationGeluSiluGolden, GeluSpecValues) {
    float vals[] = {-20.f, -10.f, -5.f, -1.f, 0.f, 1.f, 5.f, 10.f, 20.f};
    float y[9];
    harness::golden::activation_gelu_silu::compute(vals, y, 9, ActivationType::GELU);
    EXPECT_NEAR(y[4], 0.0f, 1e-5);   // GELU(0) = 0
    EXPECT_GT(y[8], 15.0f);            // GELU(20) ~ 20
}

TEST(ActivationGeluSiluGolden, SiluSpecValues) {
    float vals[] = {-20.f, -10.f, -5.f, -1.f, 0.f, 1.f, 5.f, 10.f, 20.f};
    float y[9];
    harness::golden::activation_gelu_silu::compute(vals, y, 9, ActivationType::SiLU);
    EXPECT_NEAR(y[4], 0.0f, 1e-5);   // SiLU(0) = 0
    EXPECT_GT(y[8], 19.0f);            // SiLU(20) ~ 20
}

struct ShapeCase { std::vector<int64_t> dims; std::string act_name; ActivationType act; };

class ActivationGeluSiluShapeTest : public ::testing::TestWithParam<ShapeCase> {};

TEST_P(ActivationGeluSiluShapeTest, RunnerMatchesGolden) {
    auto& tc = GetParam();
    harness::Shape shape(tc.dims);
    size_t n = shape.num_elements();
    auto x = harness::test::generate_random_float(n, kSeed);
    std::vector<float> y_golden(n), y_runner(n);

    harness::golden::activation_gelu_silu::compute(x.data(), y_golden.data(), n, tc.act);
    harness::runner::activation_gelu_silu::run(x.data(), y_runner.data(), shape, tc.act);

    auto report = harness::accuracy::CompareArrays(y_golden, y_runner,
        harness::accuracy::Tolerance{1e-5, 1e-5});
    EXPECT_TRUE(report.passed) << report.ToString() << "\nshape=" << shape.to_string()
                               << "\nseed=" << kSeed;
}

INSTANTIATE_TEST_SUITE_P(Shapes, ActivationGeluSiluShapeTest, ::testing::Values(
    ShapeCase{{128}, "gelu", ActivationType::GELU},
    ShapeCase{{1024}, "gelu", ActivationType::GELU},
    ShapeCase{{8, 512}, "silu", ActivationType::SiLU},
    ShapeCase{{2, 16, 128}, "silu", ActivationType::SiLU}
));

}  // namespace
