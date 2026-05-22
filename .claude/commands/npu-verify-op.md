# /npu-verify-op

Prove correctness of the operator by running the full test suite and producing accuracy reports.

## Usage

```
/npu-verify-op <op_name>
```

## Arguments

- `op_name` — The operator name to verify.

## Preconditions

- `ops/<op_name>/` exists with implementation files
- `tests/test_<op_name>.cpp` exists
- `.npu-harness/operators/<op_name>/SPEC.yaml` exists
- Host build directory `build/host` exists and was previously configured

## What to Read First

1. `.npu-harness/operators/<op_name>/SPEC.yaml` — Know the accuracy tolerances
2. `.npu-harness/operators/<op_name>/STATUS.md` — Know what's been done
3. `.npu-harness/operators/<op_name>/PLAN.md` — Know the test plan

## Verification Workflow

### Step 1: Clean Rebuild (Host)

Ensure the host build is fresh:

```bash
cmake --build build/host
```

If the build directory does not exist or was not configured, configure it first:

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
```

### Step 2: Run Host Tests

Run all tests matching the operator name:

```bash
ctest --test-dir build/host -R <op_name> --output-on-failure
```

Capture:
- Total tests
- Passed count
- Failed count
- Skipped count
- Failure details (full output)

### Step 3: Run Categorized Host Tests

Run tests by label to verify specific categories:

```bash
# Golden correctness tests
ctest --test-dir build/host -L golden -R <op_name> --output-on-failure

# Shape validation tests
ctest --test-dir build/host -L shape -R <op_name> --output-on-failure

# Host-only tests (should all pass)
ctest --test-dir build/host -L host -R <op_name> --output-on-failure
```

### Step 4: NPU Build and Test (if CANN available)

Check if CANN is available:

```bash
python3 scripts/detect_ascend_env.py
```

If CANN IS detected:

```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu
ctest --test-dir build/npu -R <op_name> --output-on-failure
```

Run categorized NPU tests:

```bash
ctest --test-dir build/npu -L npu -R <op_name> --output-on-failure
ctest --test-dir build/npu -L integration -R <op_name> --output-on-failure
```

If CANN is NOT detected:
- Skip NPU tests entirely
- Record in RESULTS.md: `npu_tests: skipped — CANN not detected`
- This is NOT a failure. Host-only verification is sufficient.

If CANN is detected but no NPU device is visible:
- Attempt to build (the kernel should compile)
- NPU tests will skip at runtime with `GTEST_SKIP()` for `No usable Ascend NPU device`
- Record in RESULTS.md: `npu_tests: skipped — No visible NPU device`

### Step 5: Accuracy Verification

Run the harness accuracy verification:

```bash
python3 scripts/harness.py verify <op_name>
```

This should produce a detailed accuracy comparison between NPU output (if available) and CPU golden.

The accuracy report must include:
- dtype
- shape
- element count
- absolute tolerance used
- relative tolerance used
- maximum absolute error
- maximum relative error
- index of maximum absolute error
- index of maximum relative error
- mismatch count
- first 10 mismatches (index, expected, actual, abs_error, rel_error)
- PASS or FAIL verdict

If the harness CLI doesn't yet support `verify`, do the comparison manually by reading the test output from Step 2.

### Step 6: Produce Accuracy JSON

Create `.npu-harness/runs/<timestamp>/accuracy.json`:

```json
{
  "operator": "<op_name>",
  "timestamp": "<ISO 8601>",
  "host": {
    "configure": "success",
    "build": "success",
    "tests_total": <N>,
    "tests_passed": <N>,
    "tests_failed": <N>,
    "tests_skipped": <N>
  },
  "npu": {
    "available": true,
    "configure": "success",
    "build": "success",
    "tests_total": <N>,
    "tests_passed": <N>,
    "tests_failed": <N>,
    "tests_skipped": <N>,
    "skip_reason": null
  },
  "accuracy": {
    "dtype": "<dtype>",
    "shape": "<shape>",
    "num_elements": <N>,
    "abs_tolerance": <value>,
    "rel_tolerance": <value>,
    "max_abs_error": <value>,
    "max_rel_error": <value>,
    "max_abs_index": <index>,
    "max_rel_index": <index>,
    "mismatch_count": <N>,
    "passed": true
  }
}
```

Use the timestamp format: `YYYYMMDD_HHMMSS`.

### Step 7: Produce RESULTS.md

Write `.npu-harness/operators/<op_name>/RESULTS.md`:

```markdown
# Results: <op_name>

**Date:** <today>
**Verifier:** Claude Code Harness Agent

## Build

| Mode | Configure | Build | Status |
|------|-----------|-------|--------|
| host | success | success | PASS |
| npu  | <success|skipped> | <success|skipped> | <PASS|SKIP> |

## Host Tests

| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| golden | <N> | <N> | <N> | <N> |
| shape | <N> | <N> | <N> | <N> |
| tiling | <N> | <N> | <N> | <N> |
| host (all) | <N> | <N> | <N> | <N> |

## NPU Tests

| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| npu | <N> | <N> | <N> | <N> |
| integration | <N> | <N> | <N> | <N> |

<If skipped:>
**Skip reason:** <reason>

## Accuracy

| Metric | Value |
|--------|-------|
| Dtype | <dtype> |
| Max absolute error | <value> |
| Max relative error | <value> |
| Mismatch count | <N> |
| Passed | <YES|NO> |

<If failed:>
### First 10 Mismatches
| Index | Expected | Actual | Abs Error | Rel Error |
|-------|----------|--------|-----------|-----------|
| <i> | <v> | <v> | <v> | <v> |
...

## Verdict

- [x] Host build and tests: <PASS|FAIL>
- [x] CPU golden correctness: <PASS|FAIL>
- [x] Shape validation: <PASS|FAIL>
- [x] NPU correctness (if available): <PASS|FAIL|SKIP>
- [x] Accuracy within tolerance: <PASS|FAIL>

## Overall

**<PASS | FAIL | PARTIAL>**

<If PARTIAL or FAIL, explain exactly what is missing or wrong.>
```

### Step 8: Update STATUS.md

Update `.npu-harness/operators/<op_name>/STATUS.md`:

```markdown
| host_tests | pass | <N>/<N> passed |
| npu_tests | <pass|fail|skip> | <details> |
| accuracy | <pass|fail> | <summary> |
| verify | done | |
```

## Failure Analysis

### If Host Tests Fail

This is a blocking failure. The operator is not correct on CPU.

1. Identify which tests failed.
2. Determine if the golden, shape validation, or tiling is wrong.
3. Fix the root cause (return to `/npu-execute-op`).
4. Re-run verification.
5. Do NOT continue to NPU testing until host tests pass.

### If NPU Tests Fail

1. Compare NPU output to golden for the failing cases.
2. Check if the failure is:
   - **Exact/deterministic:** Bug in kernel logic, tiling, or data movement
   - **Tolerance:** The tolerance needs adjustment (rare — prefer fixing accuracy)
   - **Specific shapes:** Bug in tail handling, block boundaries, or alignment
   - **Specific dtypes:** Dtype conversion or accumulation issue
3. Fix the root cause (return to `/npu-execute-op` Step 5).
4. Do NOT claim "acceptable" without understanding why.

### If Accuracy Fails

1. Check the first 10 mismatches.
2. Look for pattern:
   - First element wrong: initialization bug
   - Last elements wrong: tail handling
   - Every Nth element wrong: stride or tiling
   - All wrong: fundamental compute error
   - Randomly wrong: race condition or uninitialized memory
3. Fix and re-verify.

## Rules

1. Always rebuild before testing. Do not rely on stale build artifacts.
2. Host tests MUST pass. NPU test failures do not block host verification success, but they DO block the NPU accuracy verdict.
3. If CANN is unavailable, NPU tests are skipped with a clear message. This is not a failure.
4. If NPU device is unavailable, NPU runtime tests are skipped with `GTEST_SKIP()`. This is not a failure.
5. Record ALL results, including skipped tests and their reasons.
6. Never claim accuracy passes if mismatches exceed tolerance.
7. Never suppress failure output. The full test output must be included or summarized in RESULTS.md.

## Output

After verification, print a concise summary:

```
Verification complete for <op_name>.

Host:  <PASS|FAIL> (<N>/<M> tests passed)
NPU:   <PASS|FAIL|SKIP> (<reason if skip>)
Accuracy: <PASS|FAIL>

Results:   .npu-harness/operators/<op_name>/RESULTS.md
Accuracy:  .npu-harness/runs/<timestamp>/accuracy.json

Next command: /npu-tune-op <op_name> (if performance required)
              /npu-ship-op <op_name> (if no performance requirement)
```

If verification failed, the output must clearly state what failed and the next step to fix it. Do not proceed to tuning or shipping until verification passes.
