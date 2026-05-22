# verification-before-completion

## Skill Name

`verification-before-completion` -- Verification Before Completion

## Trigger Description

You are about to declare a task, operator, or phase complete. Before saying "done," "passing," "fixed," or "ready to ship," you MUST run this skill. It is a mandatory gate that prevents premature or evidence-free completion claims.

## When to Use

- Before claiming any work is "done" or "complete."
- Before committing changes.
- Before creating a pull request.
- Before running `npu-shipping` or `/npu-ship-op`.
- After implementing a feature, fix, or optimization.
- After a debugging session that claims to have resolved the issue.
- At the end of any operator workflow phase (spec, implement, test, benchmark, tune).
- Before responding to the user with a success claim.

## When Not to Use

- The work is intentionally incomplete and you are asking for guidance on next steps.
- You are in the middle of a multi-step implementation and will verify at the end.
- You are diagnosing a failure (use `systematic-debugging` instead).
- You are gathering information, not making claims.
- The user explicitly asks you to skip verification (rare -- record this request).

## Required Inputs

1. **The claims you intend to make** -- explicitly list what you believe to be true (e.g., "all host tests pass", "kernel compiles without errors", "accuracy meets fp16 tolerance").
2. **The commands that supposedly prove these claims** -- every claim must be backed by a specific command that was run or will be run.
3. **OPERATOR_REGISTRY.yaml** -- to check that status fields reflect reality.
4. **STATUS.md** -- to check that per-operator status is up to date.
5. **Run logs** at `.npu-harness/runs/<timestamp>/` -- to verify commands actually ran.

## Required Outputs

1. **Verification verdict** -- PASS or FAIL, with per-check results for all 8 mandatory checks.
2. **Evidence references** -- log file paths, exit codes, and command output excerpts backing each check.
3. **Failure report** (if FAIL) -- precise description of which check(s) failed and why, so the issue can be addressed before re-verifying.
4. **Updated STATUS.md** -- reflecting the verified state of the operator after all checks pass.

## The 8 Mandatory Checks

Each check must be performed. Each check produces a PASS, FAIL, or N/A verdict. Any FAIL blocks the completion claim.

### Check 1: Commands Actually Ran

**The claim:** "I ran the build/tests/benchmarks."

**The verification:** For each claimed command, verify it was actually executed (not just planned). Evidence sources:
- Shell output captured in logs.
- Exit codes recorded.
- Timestamped run directories exist with corresponding output files.

**Anti-pattern:** "I would run `ctest --test-dir build/host` and it would pass." -- This is a plan, not evidence. Run it.

**Evidence required:** Paste the command output or reference the log file path and show the exit code.

```
Example evidence:
$ ctest --test-dir build/host -R elementwise_add --output-on-failure
...
100% tests passed, 0 tests failed out of 5
Exit code: 0
```

### Check 2: Tests Passed (Check Exit Codes)

**The claim:** "All tests pass."

**The verification:**
- Read the exit code of the test command. Exit code 0 = all tests passed.
- Do NOT rely on memory, assumptions, or previous runs.
- Check for the exact string "0 tests failed" or "100% tests passed" in CTest output.
- For GTest, verify no `[FAILED]` markers appear in the output.

**Anti-pattern:** "I remember the tests passing." -- Memory is not evidence. Run them again.

**Anti-pattern:** "The build succeeds, so the tests must pass." -- Build success does not imply test success.

**Edge case:** A test binary crashes before producing GTest output. CTest may report this as "Test failed" with no test cases executed. This IS a test failure. Do not claim tests passed.

### Check 3: Skips Are Justified

**The claim:** "Skipped tests are expected."

**The verification:** For every `[SKIPPED]` test:
- Inspect the skip message (the string passed to `GTEST_SKIP()`).
- Verify the skip reason is VALID for the current environment.
- Invalid skips: "not implemented" (when spec says required), empty skip message, skip because of a bug that was never fixed.
- Valid skips: "Ascend runtime not available" (when no NPU), "CANN 9.0.0 missing header X" (documented blocker), "feature not yet supported for this arch" (explicitly deferred).
- A test that always skips for every configuration is a coverage gap, not a passing test. Flag it.

**Verification pattern:**
```
Skipped test: NpuMatchesGolden
Skip reason: "Ascend runtime not available"
Environment: NPU build but no device visible? Check detect_ascend_env.py output.
Verdict: VALID -- no NPU device present.
```

### Check 4: Performance Target Passed or Gap Documented

**The claim:** "Performance is acceptable" or "Performance target met."

**The verification:**
- If `performance.required: true`, check `benchmark.json` for target comparison.
- Verify `target.passed: true` in the JSON.
- If `target.passed: false`, verify that the gap is documented in PERF.md with:
  - The exact target, measured value, and gap.
  - The bottleneck hypothesis.
  - Attempted optimizations.
  - Remaining options.
- If benchmarks were not run because NPU is unavailable, this is a justified skip, not a performance failure. Mark as "pending_npu" in the registry.

**Anti-pattern:** "The operator is fast enough." -- Without a benchmark against a target, this is an opinion, not evidence.

### Check 5: Generated Files Are Listed or Committed

**The claim:** "I created/updated the necessary files."

**The verification:**
- Run `git status` or list files in the operator directory.
- For each file the operator spec requires, verify it EXISTS.
- Check that files are not empty (a 0-byte stub is not a valid implementation).
- Check that generated files have the correct generated-file markers (if from the spec generator).

**Required files checklist per operator:**
```
[ ] specs/<op_name>.yaml
[ ] ops/<op_name>/CMakeLists.txt
[ ] ops/<op_name>/kernel/<op_name>_kernel.cpp
[ ] ops/<op_name>/host/<op_name>_tiling.hpp
[ ] ops/<op_name>/host/<op_name>_tiling.cpp
[ ] ops/<op_name>/runner/<op_name>_runner.hpp
[ ] ops/<op_name>/runner/<op_name>_runner.cpp
[ ] ops/<op_name>/golden/<op_name>_golden.hpp
[ ] ops/<op_name>/golden/<op_name>_golden.cpp
[ ] tests/test_<op_name>.cpp
[ ] docs/operators/<op_name>.md
```

Mark N/A for files not required by this operator category.

### Check 6: Status Files Are Updated

**The claim:** "STATUS.md reflects the current state."

**The verification:**
- Read `.npu-harness/operators/<op_name>/STATUS.md`.
- Check that each phase (intake, spec, plan, implement, build, test, benchmark, ship) has a status.
- Check that the status matches reality: if tests passed, STATUS.md must say so; if benchmarks were skipped, STATUS.md must explain why.
- Check that timestamps are recent (not stale from a previous session).

### Check 7: Operator Registry Reflects Current State

**The claim:** "OPERATOR_REGISTRY.yaml is up to date."

**The verification:**
- Read `OPERATOR_REGISTRY.yaml`.
- Check that the operator entry has correct values for:
  - `spec` path.
  - `category`.
  - `generated` (true if spec generator was used).
  - `built_host` (true if host build passed).
  - `tested_host` (true if host tests passed).
  - `built_npu` (true if NPU build passed, false/skipped otherwise).
  - `tested_npu` (true/passed, skipped with reason, or false).
  - `accuracy` (pass, fail, or pending).
  - `performance` (pass, fail, pending, pending_npu, or sanity).
- Do NOT set `built_npu: true` if the NPU build was never run.

### Check 8: No Regressions

**The claim:** "My changes do not break anything that previously worked."

**The verification:**
- Run the FULL test suite for the operator, not just the test you changed:
  ```bash
  ctest --test-dir build/host -R <op_name> --output-on-failure
  ```
- If the operator was previously passing NPU tests, run them too:
  ```bash
  ctest --test-dir build/npu -R <op_name> --output-on-failure
  ```
- If there are cross-operator dependencies, run the broader test suite:
  ```bash
  ctest --test-dir build/host -L golden --output-on-failure
  ```
- Any newly failing test that was previously passing is a regression. Do NOT claim completion.

## Step-by-Step Procedure

### Step 1: List the Claims

Explicitly state what you believe to be true. Write this down before verification:

```
Claims:
1. Host build succeeds for all operators.
2. All host golden tests pass.
3. The elementwise_add kernel compiles for NPU.
4. Accuracy tests pass within fp16 tolerances.
5. No performance target exists, so benchmarks are sanity only.
```

### Step 2: Run Verification Commands

For each claim, run the command that would prove or disprove it. Do not assume -- execute.

### Step 3: Check Each of the 8 Gates

Go through Checks 1-8 systematically. For each:
- Record PASS, FAIL, or N/A.
- For FAIL, write a precise description of what is wrong.
- For N/A, explain why it does not apply.

### Step 4: Decide

- **All checks PASS:** The completion claim is valid. Proceed to ship, commit, or report.
- **One or more checks FAIL:** The completion claim is invalid. Fix the failures before claiming completion. Report the failures to the user.
- **All failing checks are N/A with justification:** Proceed with caution; document the N/A justifications.

### Step 5: Report

Summarize the verification result:

```
Verification result: PASS / FAIL

Check 1 (Commands ran):        PASS
Check 2 (Tests passed):        PASS  (5/5 tests passed, exit 0)
Check 3 (Skips justified):     PASS  (2 skips, both due to no NPU)
Check 4 (Performance target):  N/A   (no target required)
Check 5 (Generated files):     PASS  (10/10 files present)
Check 6 (Status files):        PASS  (STATUS.md updated)
Check 7 (Registry):            PASS  (OPERATOR_REGISTRY.yaml correct)
Check 8 (No regressions):      PASS  (full suite passes)

Verdict: Ready to proceed.
```

## Must NOT Do

These behaviors invalidate the verification:

| Anti-pattern | Why it is wrong |
|---|---|
| Claiming success without running commands | Plans are not evidence. |
| Accepting zero-test passes as success | A test binary with 0 tests passes with exit 0. This is not coverage. |
| Ignoring skipped tests | Every skip must be reviewed and justified. |
| Not checking exit codes | Output may say "FAILED" but exit code was ignored. |
| Assuming previous state persists | Build artifacts may be stale; tests may have been modified. |
| Claiming "all tests pass" when only a subset ran | "I ran the one test I changed" is not "all tests pass." |
| Proceeding with a FAIL on any check | Fix the failure. Do not "accept and note." |

## Verification Checklist

Before claiming completion, verify every item:

- [ ] Check 1: Every claimed command was actually run (not just planned).
- [ ] Check 1: Command output is captured in logs or pasted in the session.
- [ ] Check 2: Test exit code is 0.
- [ ] Check 2: CTest output shows "100% tests passed" or "0 tests failed".
- [ ] Check 2: No GTest `[FAILED]` markers in output.
- [ ] Check 3: Every `GTEST_SKIP()` has a clear, valid reason string.
- [ ] Check 3: No test always-skips; each skip is environment-conditional.
- [ ] Check 4: Performance target comparison exists if `performance.required: true`.
- [ ] Check 4: Performance gap is documented if target not met.
- [ ] Check 5: All required operator files exist and are non-empty.
- [ ] Check 5: Generated file markers are present where applicable.
- [ ] Check 6: `.npu-harness/operators/<op_name>/STATUS.md` is current.
- [ ] Check 7: `OPERATOR_REGISTRY.yaml` fields match observed reality.
- [ ] Check 7: Registry does not claim NPU success if NPU build was never run.
- [ ] Check 8: Full test suite for the operator passes (not just the changed test).
- [ ] Check 8: No previously passing tests now fail.
