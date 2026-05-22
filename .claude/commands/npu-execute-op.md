# /npu-execute-op

Implement the operator according to SPEC.yaml and PLAN.md, following the RED -> GREEN -> REFACTOR cycle.

## Usage

```
/npu-execute-op <op_name>
```

## Arguments

- `op_name` — The operator name to implement.

## Preconditions

- `.npu-harness/operators/<op_name>/SPEC.yaml` — Complete and validated spec
- `.npu-harness/operators/<op_name>/PLAN.md` — Complete implementation plan
- `.npu-harness/operators/<op_name>/DECISIONS.md` — All decisions recorded
- Implementation scaffold exists under `ops/<op_name>/`
- Test file exists at `tests/test_<op_name>.cpp`

If the plan is missing or incomplete, run `/npu-plan-op <op_name>` first.

## What to Read Before Implementing

1. `.npu-harness/operators/<op_name>/SPEC.yaml` — The contract you are implementing
2. `.npu-harness/operators/<op_name>/PLAN.md` — The step-by-step plan
3. `.npu-harness/operators/<op_name>/DECISIONS.md` — Design decisions and assumptions
4. Existing similar operators under `ops/` — Code patterns and conventions
5. `include/harness/` — Available utility headers (accuracy, shape, dtype, tensor, etc.)
6. `src/` — Available utility implementations

## Workflow: RED -> GREEN -> REFACTOR

Execute each step from PLAN.md in order. For each step, follow the RED -> GREEN -> REFACTOR cycle.

### Phase 1: RED — Write Failing Tests

**Goal:** Tests exist, compile, and fail because the implementation is missing or stubbed.

1. Open `tests/test_<op_name>.cpp`.
2. Implement the test cases listed in PLAN.md Test Plan.
3. Use the `gtest-verification` skill for test structure guidance.

Each test case must include:
- A descriptive test name
- Deterministic input data (seeded random)
- Expected output computed from the mathematical definition
- Comparison using Harness accuracy utilities
- Appropriate CTest labels

Test structure:
```cpp
#include <gtest/gtest.h>
#include "harness/accuracy.hpp"
#include "harness/test_utils.hpp"

// Host golden test
TEST(<op_name>Test, BasicShape) {
    // Arrange
    std::vector<float> x = ...;
    std::vector<float> expected = ...;
    std::vector<float> actual(x.size());

    // Act
    <op_name>_golden(x.data(), actual.data(), /*params*/);

    // Assert
    auto report = harness::CompareArrays(expected, actual, tolerance);
    EXPECT_TRUE(report.passed) << report.ToString();
}

// Shape validation test
TEST(<op_name>Test, RejectInvalidShape) {
    // Invalid shape should throw, return error code, or produce empty output
}

// Invalid input test
TEST(<op_name>Test, RejectMismatchedShapes) {
    // Dimensionality mismatch should be caught
}
```

NPU tests (if CANN available):
```cpp
TEST(<op_name>NPUTest, CompareWithGolden) {
    if (!harness::ascend::HasUsableDevice()) {
        GTEST_SKIP() << "No usable Ascend NPU device";
    }
    // Run NPU path via runner, compare with CPU golden
}
```

3. Configure and build in host-only mode:
```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
```

4. Run the tests:
```bash
ctest --test-dir build/host -R <op_name> --output-on-failure
```

5. **Confirm RED:** Tests should fail because:
   - Golden function is a stub or does nothing
   - Shape validation accepts everything
   - NPU runner is not implemented

If tests pass when they should fail, the tests are insufficient. Add more rigorous assertions.

### Phase 2: GREEN — Implement in Order

For each implementation step from PLAN.md, implement the code and then run tests to confirm progress.

**Use the relevant skills for each step:**

| Step | Skill to Use |
|------|-------------|
| CPU golden | `cpu-golden-authoring` |
| Shape validation | `npu-tiling-design` (shape inference part) |
| Tiling implementation | `npu-tiling-design` |
| NPU kernel | `ascend-c-kernel-implementation` |
| Runner | `ascend-c-kernel-implementation` (launch path) |

**Step-by-step implementation order:**

#### Step 2: CPU Golden

Implement `ops/<op_name>/golden/<op_name>_golden.cpp`:

- Pure C++ reference implementation
- Use float32 for internal accumulation when appropriate
- Numerically stable algorithms (e.g., for softmax, normalization)
- Handle all dtypes listed in the spec
- No dependency on PyTorch, NumPy, or any framework

Rebuild and run host tests:
```bash
cmake --build build/host
ctest --test-dir build/host -R <op_name> --output-on-failure
```

**Confirm GREEN:** Golden tests pass. If they don't, the golden is wrong or the tests are wrong. Fix the bug before moving on.

#### Step 3: Shape Validation

Implement `ops/<op_name>/host/<op_name>_tiling.cpp`:

- Validate input rank against spec
- Validate each dimension against shape_rules
- Validate dtype compatibility
- Compute output shape
- Return clear error messages for violations

Add shape validation tests if not already in test file:
```cpp
TEST(<op_name>Test, RejectWrongRank) { ... }
TEST(<op_name>Test, RejectUnsupportedDtype) { ... }
TEST(<op_name>Test, AcceptValidShapes) { ... }
```

Rebuild and run tests. **Confirm GREEN:** Shape validation tests pass.

#### Step 4: Tiling Implementation

Implement tiling logic in `ops/<op_name>/host/<op_name>_tiling.cpp`:

- Compute tile sizes per dimension
- Compute block dimensions for kernel launch
- Enforce alignment constraints (e.g., 32-byte alignment)
- Define tail handling policy
- Define per-core work distribution
- Compute workspace size if needed

Use the `npu-tiling-design` skill for guidance.

Add tiling validation tests:
```cpp
TEST(<op_name>Test, TilingProducesValidBlocks) { ... }
TEST(<op_name>Test, TilingCoversAllElements) { ... }
TEST(<op_name>Test, TailHandling) { ... }
```

Rebuild and run tests. **Confirm GREEN:** Tiling tests pass.

#### Step 5: NPU Kernel

Implement `ops/<op_name>/kernel/<op_name>_kernel.cpp`:

- Ascend C kernel using the local CANN conventions
- Target arch: dav-2201
- Process data in tiles (not whole tensors at once)
- Safe tail handling (no out-of-bounds access)
- Proper dtype conversions
- Use `<<<>>>` launch syntax

Use the `ascend-c-kernel-implementation` skill for guidance.

If CANN is NOT available:
- Write the kernel code based on known Ascend C patterns
- Mark the kernel as `#ifdef HARNESS_ENABLE_ASCEND` guarded
- Do NOT claim NPU tests pass without actual NPU execution
- Record in BLOCKERS.md: `"NPU kernel written but not compiled — CANN not available"`

If CANN IS available, build the NPU target:
```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu
```

#### Step 6: Runner

Implement `ops/<op_name>/runner/<op_name>_runner.cpp`:

- Allocate device memory
- Copy input data to device
- Launch kernel with `<<<>>>` syntax (blocks, stream, device pointers, scalar params, tiling params)
- Synchronize stream
- Copy output data from device
- Free device memory (RAII)
- Handle errors gracefully

Runner API should be:
```cpp
void <op_name>_runner(
    const float* x,       // host input
    float* y,             // host output
    const <op_name>_tiling_params& tp,  // tiling params
    int64_t num_elements,
    // ... other params
);
```

#### Step 7: Run Full Test Suite

```bash
# Host tests (always)
cmake --build build/host
ctest --test-dir build/host -R <op_name> --output-on-failure

# NPU tests (if CANN available)
cmake --build build/npu
ctest --test-dir build/npu -R <op_name> --output-on-failure
```

**Confirm GREEN:** All tests pass (host tests always; NPU tests pass or skip cleanly).

### Phase 3: REFACTOR

After all tests pass, review the code for:

1. **Duplication:** Any repeated logic between golden and kernel?
2. **Clarity:** Are variable names descriptive? Is the flow obvious?
3. **Safety:** Are there any potential out-of-bounds accesses? Unhandled error paths?
4. **Performance:** Any obvious algorithmic inefficiencies? (Do NOT optimize prematurely; only fix clear waste.)
5. **Consistency:** Does the code follow the same patterns as existing operators?

Refactor only if there is a clear improvement. Do not refactor just to make it "nicer." Every refactor must be followed by re-running the full test suite.

### Step 8: Update STATUS.md

After implementation is complete, update `.npu-harness/operators/<op_name>/STATUS.md`:

```markdown
| golden | done | |
| shape_validation | done | |
| tiling | done | |
| npu_kernel | done | <or 'stubbed — CANN not available'> |
| runner | done | <or 'stubbed — CANN not available'> |
| host_tests | pass | |
| npu_tests | <pass|skip|pending> | <reason if skip> |
```

## Error Handling During Implementation

### Build Failure

If build fails:
1. Read the full error output.
2. Identify the first error (subsequent errors may be cascading).
3. Fix ONLY the first error.
4. Rebuild.
5. Repeat until build succeeds.

Use the `systematic-debugging` skill for build failures.

### Test Failure

If tests fail:
1. Read the test failure output carefully.
2. Identify: is the golden wrong, the kernel wrong, or the test wrong?
3. Golden wrong: fix the golden, verify math against spec.
4. Kernel wrong: compare with golden step by step on small shapes.
5. Test wrong: fix the test expectations.
6. Rerun.

Use the `systematic-debugging` skill for test failures.

### Accuracy Failure

If golden passes but NPU-kernel-vs-golden comparison fails:
1. Print first 10 mismatches with indices, expected, actual, abs_error, rel_error.
2. Identify pattern: all wrong (fundamental bug), specific indices wrong (edge case), large values only (dtype overflow), small values only (underflow).
3. Fix root cause.
4. Rerun accuracy comparison.

## Rules

1. Follow PLAN.md step order. Do not skip steps or reorder them.
2. CPU golden always before NPU kernel. No exceptions.
3. Tests always before implementation (RED before GREEN).
4. Run tests after EVERY implementation step. Do not batch implementation and then run tests at the end.
5. If CANN is unavailable, write the kernel and runner code but clearly mark it as "not compiled/tested on NPU." Do NOT claim NPU tests pass.
6. If a test is flaky, fix the nondeterminism (add a fixed seed). Do not ignore flaky tests.
7. Every failure must be understood before fixing. Do not randomly change code hoping it works.
8. Update STATUS.md after each phase completes. Do not wait until the end.

## Output

After implementation, print a summary:

```
Implementation complete for <op_name>.

Results:
  golden:         <pass/fail>
  shape_validation: <pass/fail>
  tiling:         <pass/fail>
  npu_kernel:     <pass/skip/stubbed>
  runner:         <pass/skip/stubbed>
  host_tests:     <N passed, M failed, K skipped>
  npu_tests:      <N passed, M failed, K skipped>

Files modified:
  ops/<op_name>/golden/<op_name>_golden.cpp
  ops/<op_name>/host/<op_name>_tiling.cpp
  ops/<op_name>/kernel/<op_name>_kernel.cpp
  ops/<op_name>/runner/<op_name>_runner.cpp
  tests/test_<op_name>.cpp
  .npu-harness/operators/<op_name>/STATUS.md

Next command: /npu-verify-op <op_name>
```

If any step is blocked, produce a precise blocker report instead:

```
Implementation BLOCKED for <op_name>.

Completed steps: 1, 2, 3
Blocked at step: 4 (Tiling)
Blocker: <specific missing dependency, API, or information>
Recommendation: <concrete next action>
```

Do not claim success if tests fail or were skipped without justification.
