# gtest-verification

## Skill Name

`gtest-verification` -- GTest and CTest Verification

## Trigger Description

The user requests writing tests for an operator. This covers GTest test files, CTest registration, test data generation, golden comparison, and edge case coverage. The tests are the correctness gate between implementation and shipping.

## When to Use

- A SPEC.yaml exists with `tests.shapes` populated, and the CPU golden is implemented.
- The user says "write tests for <op>" or "add tests for <op>".
- `/npu-execute-op <op_name>` has reached the testing step.
- `/npu-verify-op <op_name>` is invoked to run and validate tests.
- A bug was fixed and regression tests are needed.
- A new dtype or shape variant needs test coverage.

## When Not to Use

- No SPEC.yaml exists -- run `npu-spec-authoring` first.
- No CPU golden exists -- run `cpu-golden-authoring` first.
- The task is to benchmark, not test correctness -- use `performance-benchmarking`.
- Tests already exist and pass; you are adding a new operator -- create new test file, not modify existing.
- A single test is failing and you are debugging -- use `systematic-debugging` to find and fix the cause.

## Required Inputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` -- provides shapes, dtypes, tolerances, edge cases, invalid cases, and the deterministic seed.
2. **CPU golden** at `ops/<op_name>/golden/<op_name>_golden.{hpp,cpp}` -- the reference implementation to compare against.
3. **NPU runner** at `ops/<op_name>/runner/<op_name>_runner.{hpp,cpp}` (if NPU tests are being written).
4. **Harness test utilities** at `include/harness/test_utils.hpp` and `include/harness/accuracy.hpp`.
5. **OPERATOR_REGISTRY.yaml** -- for determining which CTest labels to apply.

## Required Outputs

1. **`tests/test_<op_name>.cpp`** -- the GTest test file.
2. **Updated `tests/CMakeLists.txt`** -- registering the new test target.
3. **CTest labels** applied to the test target.
4. **Updated STATUS.md** -- reflecting test results.

## Mandatory Requirements

### Requirement 1: Shape Coverage

Every shape listed in SPEC.yaml `tests.shapes` must have at least one test case. If the spec lists 5 shapes, there must be at least 5 distinct shape test configurations. Use `INSTANTIATE_TEST_SUITE_P` or a test loop over shapes.

### Requirement 2: Dtype Coverage

Every dtype listed in SPEC.yaml `inputs[].dtype` must be tested. If the spec lists `[float16, float32]`, there must be tests for both dtypes.

### Requirement 3: Edge Case Coverage

Every edge case listed in SPEC.yaml `tests.edge_cases` must have a test:

- Zeros input
- Ones input
- Large values (near dtype max)
- Small values (near dtype min)
- nan and inf propagation (if applicable)
- Singleton dimensions (all dims = 1)
- Non-power-of-2 dimensions (e.g., 17, 63, 257)
- Large dimensions (e.g., 4096, 8192) for reduction operators
- Empty tensors (if supported)

### Requirement 4: Invalid Input Rejection

Every invalid case listed in SPEC.yaml `tests.invalid_cases` must have a test that verifies the operator REJECTS the input (throws exception, returns error code, or otherwise fails gracefully).

### Requirement 5: Golden Comparison with Correct Tolerances

Every comparison between NPU output and CPU golden must use the tolerances from SPEC.yaml `accuracy`. Use the harness comparison utility:

```cpp
#include "harness/accuracy.hpp"

harness::accuracy::Tolerance tol{
    spec_accuracy.<dtype>.abs,
    spec_accuracy.<dtype>.rel
};

auto report = harness::accuracy::CompareArrays(
    expected_span, actual_span, tol
);

EXPECT_TRUE(report.passed) << report.ToString();
```

If `CompareArrays` is not yet available, implement equivalent comparison logic inline.

### Requirement 6: GTest Skip for NPU Unavailable

Every test that requires NPU hardware must skip gracefully:

```cpp
TEST_F(MyOpTest, NpuMatchesGolden) {
    if (!harness::ascend::IsAscendRuntimeAvailable()) {
        GTEST_SKIP() << "Ascend runtime not available; skipping NPU test.";
    }
    if (!harness::ascend::HasUsableDevice()) {
        GTEST_SKIP() << "No usable Ascend device; skipping NPU test.";
    }
    // ... actual NPU test
}
```

Host-only tests (CPU golden correctness) must NEVER skip due to missing NPU.

### Requirement 7: CTest Labels

Every test must be registered with CTest labels. The labels must include:

| Label | When to apply |
|---|---|
| `host` | Test runs in host-only mode (always applied). |
| `golden` | Test exercises the CPU golden reference. |
| `npu` | Test requires NPU hardware. |
| `<category>` | One of: `elementwise`, `reduction`, `layout`, `matmul`, `normalization`, `attention`, `fusion`, `llm`. |
| Additional labels as needed | `shape`, `tiling`, `integration`, `benchmark`, `slow`, `flash_attention`. |

Apply labels in CMakeLists.txt:

```cmake
add_test(NAME <op_name>_<test_name> COMMAND <test_target>)
set_tests_properties(<op_name>_<test_name> PROPERTIES LABELS "host;golden;elementwise")
```

### Requirement 8: Deterministic Seed Printed on Failure

All randomized tests must use the deterministic seed `20260522` from the harness. Print the seed when a test fails:

```cpp
constexpr uint32_t kDeterministicSeed = 20260522;

// In test:
std::mt19937 rng(kDeterministicSeed);

// In EXPECT/ASSERT failure messages:
EXPECT_TRUE(report.passed)
    << "Seed: " << kDeterministicSeed
    << "\n" << report.ToString();
```

### Requirement 9: Use harness::accuracy::CompareArrays

The canonical comparison function is `harness::accuracy::CompareArrays`. When available, use it. When not available, implement an equivalent function that reports:
- Max absolute error and index.
- Max relative error and index.
- Mismatch count.
- First N mismatches with expected/actual/abs_error/rel_error.

## Step-by-Step Procedure

### Step 1: Read the SPEC.yaml Test Section

Extract from `specs/<op_name>.yaml`:
- `tests.shapes` -- concrete shape list.
- `tests.edge_cases` -- edge case descriptions.
- `tests.invalid_cases` -- invalid input descriptions.
- `tests.deterministic_seed` -- seed value (should be 20260522).
- `accuracy.<dtype>.abs` and `accuracy.<dtype>.rel` -- tolerances per dtype.

### Step 2: Create the Test File

Create `tests/test_<op_name>.cpp` with this structure:

```cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "harness/accuracy.hpp"
#include "harness/dtype.hpp"
#include "harness/shape.hpp"
#include "harness/test_utils.hpp"
#include "harness/ascend_runtime.hpp"

// Golden
#include "ops/<op_name>/golden/<op_name>_golden.hpp"

// Runner (for NPU tests)
#ifdef HARNESS_ENABLE_ASCEND
#include "ops/<op_name>/runner/<op_name>_runner.hpp"
#endif

namespace {

constexpr uint32_t kSeed = 20260522;

class <OpName>Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }
};

// --- Host/Golden Tests ---

TEST_F(<OpName>Test, GoldenProducesExpectedOutput) {
    // Verify golden against hand-computed values
}

TEST_F(<OpName>Test, GoldenHandlesAllShapes) {
    // Loop over spec shapes
}

// --- Dtype Tests ---

TEST_F(<OpName>Test, Float32GoldenMatchesExpected) { /* ... */ }
TEST_F(<OpName>Test, Float16GoldenMatchesExpected) { /* ... */ }

// --- Edge Case Tests ---

TEST_F(<OpName>Test, ZerosInput) { /* ... */ }
TEST_F(<OpName>Test, OnesInput) { /* ... */ }
TEST_F(<OpName>Test, NonPowerOfTwoShape) { /* ... */ }

// --- Invalid Input Tests ---

TEST_F(<OpName>Test, RejectsDimensionMismatch) { /* ... */ }

// --- NPU Tests (guarded) ---

#ifdef HARNESS_ENABLE_ASCEND
TEST_F(<OpName>Test, NpuMatchesGolden) {
    if (!harness::ascend::IsAscendRuntimeAvailable()) {
        GTEST_SKIP() << "Ascend runtime not available.";
    }
    // ... compare NPU output to golden
}
#endif

}  // namespace
```

### Step 3: Implement the Test Data Generator

Write a helper that generates deterministic test data:

```cpp
template <typename T>
std::vector<T> GenerateTestData(size_t count, uint32_t seed, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<T> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(dist(rng));
    }
    return data;
}
```

Use a fresh seed offset for each test to avoid correlated data:

```cpp
auto data_a = GenerateTestData<float>(count, kSeed + 0);
auto data_b = GenerateTestData<float>(count, kSeed + 1);
```

### Step 4: Implement Per-Shape Test Cases

For each shape in `tests.shapes`, create a test or parameterized test:

```cpp
TEST_F(<OpName>Test, ShapeCorrectness) {
    struct TestShape { std::vector<int64_t> dims; };
    std::vector<TestShape> shapes = {
        {{1}},
        {{1024}},
        {{3, 7}},
        {{4, 16, 64}},
        {{2, 3, 32, 128}},
    };
    for (const auto& sh : shapes) {
        // ... run golden and verify output shape
    }
}
```

### Step 5: Implement Invalid Input Tests

For each invalid case, verify that the operator gracefully rejects:

```cpp
TEST_F(<OpName>Test, RejectsMismatchedShapes) {
    std::vector<float> a(100), b(200);
    EXPECT_THROW(
        harness::golden::<op_name>::Compute(a.data(), b.data(), /*mismatched shapes*/),
        std::invalid_argument
    );
}
```

### Step 6: Register Tests in CMakeLists.txt

Update `tests/CMakeLists.txt`:

```cmake
add_executable(test_<op_name> test_<op_name>.cpp)
target_link_libraries(test_<op_name> PRIVATE
    GTest::gtest_main
    harness_core
    <op_name>_golden
    # <op_name>_runner  # only if NPU tests are compiled
)

# CTest registration with labels
include(GoogleTest)
gtest_discover_tests(test_<op_name>
    PROPERTIES
        LABELS "host;golden;<category>"
)

# NPU-specific test (if applicable)
if(HARNESS_ENABLE_ASCEND)
    target_link_libraries(test_<op_name> PRIVATE <op_name>_runner)
    target_compile_definitions(test_<op_name> PRIVATE HARNESS_ENABLE_ASCEND)
endif()
```

### Step 7: Run Host Tests

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF
cmake --build build/host
ctest --test-dir build/host -R <op_name> --output-on-failure
```

### Step 8: Run NPU Tests (if available)

```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201
cmake --build build/npu
ctest --test-dir build/npu -R <op_name> --output-on-failure
```

### Step 9: Verify Skip Behavior

If NPU is unavailable, confirm that NPU tests show `[SKIPPED]` with a clear message, not `[FAILED]`.

## Verification Checklist

Before declaring tests complete, verify each item:

- [ ] Test file `tests/test_<op_name>.cpp` exists.
- [ ] Every shape from SPEC.yaml `tests.shapes` has at least one test case.
- [ ] Every dtype from SPEC.yaml `inputs[].dtype` is tested.
- [ ] Edge case tests exist: zeros, ones, large values, small values, nan/inf.
- [ ] Non-power-of-2 dimension tests exist.
- [ ] Singleton dimension tests exist.
- [ ] Large dimension tests exist (for reductions, matmuls, attention).
- [ ] Invalid input rejection tests exist.
- [ ] Golden comparison uses correct per-dtype tolerances from SPEC.yaml.
- [ ] NPU tests skip with `GTEST_SKIP()` when CANN/NPU unavailable (not `EXPECT` or `ASSERT`).
- [ ] Host/golden tests never skip due to missing NPU.
- [ ] CTest labels applied: `host`, `golden`, `<category>`, plus `npu` if applicable.
- [ ] Deterministic seed `20260522` is used for all randomized data generation.
- [ ] Seed is printed in failure messages.
- [ ] `harness::accuracy::CompareArrays` (or equivalent) is used for comparison.
- [ ] `tests/CMakeLists.txt` is updated with the new test target.
- [ ] Host build succeeds.
- [ ] All host tests pass (`ctest --test-dir build/host -R <op_name>`).
- [ ] NPU tests pass or skip cleanly.
- [ ] Zero tests that pass without checking values (no `EXPECT_TRUE(true)` as a placeholder).
- [ ] STATUS.md is updated with test results.
