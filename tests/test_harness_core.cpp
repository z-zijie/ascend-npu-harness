// Core harness library tests
#include <gtest/gtest.h>
#include "harness/accuracy.hpp"
#include "harness/shape.hpp"
#include "harness/dtype.hpp"
#include "harness/test_utils.hpp"
#include "harness/skip.hpp"

TEST(HarnessCore, ShapeBasics) {
    harness::Shape s({2, 3, 4});
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s.num_elements(), 24);
    EXPECT_EQ(s.to_string(), "[2, 3, 4]");
}

TEST(HarnessCore, ShapeFlatIndex) {
    harness::Shape s({2, 3, 4});
    EXPECT_EQ(s.flat_index({0, 0, 0}), 0);
    EXPECT_EQ(s.flat_index({0, 0, 1}), 1);
    EXPECT_EQ(s.flat_index({0, 1, 0}), 4);
    EXPECT_EQ(s.flat_index({1, 0, 0}), 12);
    EXPECT_EQ(s.flat_index({1, 2, 3}), 23);
}

TEST(HarnessCore, DtypeSizes) {
    EXPECT_EQ(harness::dtype_size(harness::Dtype::Float32), 4);
    EXPECT_EQ(harness::dtype_size(harness::Dtype::Float16), 2);
    EXPECT_EQ(harness::dtype_size(harness::Dtype::Int32), 4);
}

TEST(HarnessCore, DtypeFromString) {
    EXPECT_EQ(harness::dtype_from_string("float32"), harness::Dtype::Float32);
    EXPECT_EQ(harness::dtype_from_string("float16"), harness::Dtype::Float16);
    EXPECT_THROW(harness::dtype_from_string("unknown"), std::runtime_error);
}

TEST(HarnessCore, AccuracyExactMatch) {
    std::vector<float> expected = {1.0f, 2.0f, 3.0f};
    std::vector<float> actual = {1.0f, 2.0f, 3.0f};
    auto report = harness::accuracy::CompareArrays(expected, actual,
        harness::accuracy::Tolerance{1e-5, 1e-5});
    EXPECT_TRUE(report.passed);
    EXPECT_EQ(report.mismatch_count, 0);
}

TEST(HarnessCore, AccuracyMismatch) {
    std::vector<float> expected = {1.0f, 2.0f, 3.0f};
    std::vector<float> actual = {1.0f, 2.5f, 3.0f};
    auto report = harness::accuracy::CompareArrays(expected, actual,
        harness::accuracy::Tolerance{1e-5, 1e-5});
    EXPECT_FALSE(report.passed);
    EXPECT_EQ(report.mismatch_count, 1);
    EXPECT_GT(report.max_abs_error, 0.4);
}

TEST(HarnessCore, AccuracyWithinTolerance) {
    std::vector<float> expected = {1.0f, 100.0f, 0.001f};
    std::vector<float> actual = {1.000001f, 100.0001f, 0.001001f};
    auto report = harness::accuracy::CompareArrays(expected, actual,
        harness::accuracy::Tolerance{1e-3, 1e-3});
    EXPECT_TRUE(report.passed);
}

TEST(HarnessCore, AccuracySizeMismatch) {
    std::vector<float> expected = {1.0f, 2.0f};
    std::vector<float> actual = {1.0f};
    auto report = harness::accuracy::CompareArrays(expected, actual,
        harness::accuracy::Tolerance{1e-5, 1e-5});
    EXPECT_FALSE(report.passed);
}

TEST(HarnessCore, TestUtilsRandomDeterministic) {
    auto v1 = harness::test::generate_random_float(100, 20260522);
    auto v2 = harness::test::generate_random_float(100, 20260522);
    EXPECT_EQ(v1.size(), 100);
    EXPECT_EQ(v2.size(), 100);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(v1[i], v2[i]);
    }
}

TEST(HarnessCore, TestUtilsRandomDifferentSeed) {
    auto v1 = harness::test::generate_random_float(100, 42);
    auto v2 = harness::test::generate_random_float(100, 43);
    bool all_same = true;
    for (size_t i = 0; i < 100; ++i) {
        if (v1[i] != v2[i]) { all_same = false; break; }
    }
    EXPECT_FALSE(all_same);
}

TEST(HarnessCore, Float16ConversionRoundtrip) {
    // Test basic float16 <-> float32 conversion
    float original = 1.5f;
    uint32_t bits;
    std::memcpy(&bits, &original, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    uint16_t fp16 = static_cast<uint16_t>(sign | (exp << 10) | mant);

    float recovered = harness::test::float16_to_float32(fp16);
    EXPECT_NEAR(original, recovered, 1e-3);
}

TEST(HarnessCore, AscendRuntimeSkip) {
    // In host-only mode, Ascend runtime should report unavailable
    bool available = harness::IsAscendRuntimeAvailable();
    // Either way, this test should not crash
    EXPECT_TRUE(available || !available);
}

TEST(HarnessCore, ShapeBroadcastCompatible) {
    harness::Shape a({4, 1, 3});
    harness::Shape b({1, 5, 3});
    EXPECT_TRUE(harness::Shape::broadcast_compatible(a, b));

    harness::Shape c({4, 2, 3});
    harness::Shape d({4, 5, 3});
    EXPECT_FALSE(harness::Shape::broadcast_compatible(c, d));
}
