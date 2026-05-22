# systematic-debugging

## Skill Name

`systematic-debugging` -- Systematic Debugging for NPU Operators

## Trigger Description

Something is broken. A build fails, a test fails, accuracy is out of tolerance, a benchmark crashes, or the performance is unexpectedly poor. This skill provides a disciplined debugging methodology that follows a reproduce -> isolate -> explain -> fix -> rerun -> record cycle.

## When to Use

- A build command exits with non-zero status.
- One or more CTest tests fail (FAILED, not SKIPPED).
- An accuracy comparison reports mismatches exceeding tolerances.
- A benchmark crashes, hangs, or produces nonsensical results (e.g., 0 us, negative time).
- A kernel produces NaN, Inf, or garbage output.
- The user says "debug <op>" or "fix <op>" or "something is wrong with <op>".
- A previously passing test starts failing (regression).

## When Not to Use

- The user is asking a design question ("how should I implement X?") -- use the relevant authoring skill.
- The user is adding a new feature, not fixing a bug -- use `ascend-c-kernel-implementation` or similar.
- The user is running benchmarks to measure performance (not debugging) -- use `performance-benchmarking`.
- Tests are skipped because NPU is unavailable -- this is expected behavior, not a bug.
- The user is asking about status or progress -- read STATE.md and OPERATOR_REGISTRY.yaml directly.

## Required Inputs

1. **The failing command** -- exactly as run by the user or the harness.
2. **The failure output** -- stdout, stderr, exit code, and any log files.
3. **SPEC.yaml** at `specs/<op_name>.yaml` -- the authoritative spec to check against.
4. **Relevant source files** -- kernel, tiling, golden, runner, test file -- depending on the failure type.
5. **Build logs** if the failure is a build error.
6. **CTest logs** if the failure is a test error.
7. **Previous passing state** -- when did this last work? What changed?

## Required Outputs

1. **Root cause analysis** -- a precise explanation of the defect in terms of code/spec mismatch, recorded in the session or in BLOCKERS.md.
2. **Minimal fix** -- one targeted code change per debug iteration, with a comment if non-obvious.
3. **Rerun evidence** -- the previously failing command re-executed and now passing, with output captured to `.npu-harness/runs/<timestamp>/`.
4. **Updated status files** -- STATUS.md, BLOCKERS.md, and/or DECISIONS.md updated with findings, fix, and resolution.
5. **Regression check** -- full operator test suite run to verify no new failures were introduced.

## Core Methodology: The 6-Step Debugging Cycle

This methodology is mandatory. Do not skip steps. Do not combine steps. Each step produces evidence.

### Step 1: REPRODUCE

Re-run the EXACT failing command. Capture ALL output.

```
$ cmake --build build/host 2>&1 | tee .npu-harness/runs/<timestamp>/build.log
$ ctest --test-dir build/host -R <op_name> --output-on-failure 2>&1 | tee .npu-harness/runs/<timestamp>/ctest.log
```

Requirements:
- Use the exact command that failed, not a modified version.
- Capture stdout AND stderr.
- Record the exit code (`echo $?`).
- Write output to a timestamped log file.
- Note the environment: working directory, env vars, cmake cache values.

If the failure is not reproducible (the command now succeeds), investigate what changed. An intermittent failure is a bug in itself -- record it as such.

### Step 2: ISOLATE

Identify the specific component responsible for the failure. Narrow the scope from "the whole operator is broken" to "the tiling function computes the wrong block count" or "the kernel tail loop writes past the end of the buffer."

**For build failures:**
- Which file fails to compile?
- Which line number?
- What is the error message?
- Is it a missing header, syntax error, type mismatch, or linker error?

**For test failures:**
- Which specific test case fails? (Not the whole test suite -- the specific `TEST_F` or `TEST` name.)
- What assertion fails? `EXPECT_EQ`, `EXPECT_TRUE`, `EXPECT_NEAR`?
- What are the actual vs. expected values?
- Is it a golden comparison failure, a shape mismatch, or a runtime error?

**For accuracy failures:**
- Which dtype? Which shape?
- What is the max absolute error and at which index?
- What is the max relative error and at which index?
- Is the error localized (a few elements) or global (all elements)?
- Do the first few mismatches suggest a pattern (off-by-one, wrong stride, missing scale)?

**For performance anomalies:**
- Which shape and dtype?
- Is it consistently slow or sporadically slow?
- Is the first iteration much slower than the rest? (Cold-start issue.)
- Is the NPU utilization low? (Check profiling data if available.)

**For crashes/hangs:**
- Does it crash immediately or after some iterations?
- Which line? (Use debug output, `printf` tracing, or `GDB` if available.)
- Does changing the shape (smaller, larger, different alignment) change the behavior?

Write a one-sentence isolation statement:
```
The failure is in <component> at <location>. The symptom is <observed behavior>.
```

### Step 3: EXPLAIN

Articulate the ROOT CAUSE in terms of code/spec mismatch, not symptoms. A root cause explanation must answer:

1. What does the code DO? (Observed behavior.)
2. What SHOULD the code do? (Expected behavior per spec.)
3. WHY does the code do the wrong thing? (The actual defect.)

Bad explanations (symptom-level, not actionable):
- "The test fails because the values don't match." (Restates the symptom.)
- "The kernel has a bug." (Too vague to fix.)
- "The golden is wrong." (Does not say why or where.)

Good explanations (root cause, actionable):
- "The kernel computes `output = alpha * x + beta` but the spec defines `output = alpha * x + beta * y`. The `beta * y` term is missing because the kernel uses only `beta` as a scalar addend without multiplying by `y`."
- "The tiling function computes `num_blocks = total_elements / tile_size` using integer division, which drops the remainder when `total_elements % tile_size != 0`. The kernel then processes only `num_blocks * tile_size` elements, skipping the tail. The fix is to add `(total_elements + tile_size - 1) / tile_size` or explicitly handle the tail."
- "The golden accumulates in fp16, losing precision for K=4096. The max absolute error is 0.5 at index 0 because the accumulator saturates. The fix is to use fp32 accumulation in the golden as required by REQ-3 of `cpu-golden-authoring`."

### Step 4: FIX

Change ONE root cause at a time. Do NOT fix multiple things in a single iteration.

Rules:
- Change only the lines directly related to the root cause identified in Step 3.
- Do not refactor unrelated code during a debugging fix.
- Do not add "defensive" checks that mask the root cause without understanding it.
- Do not change tolerances to make a failing accuracy test pass unless analysis proves the spec tolerance was incorrect.
- Write the fix as a minimal diff.
- Add a comment near the fix referencing the root cause if it is non-obvious.

After the fix, state what you changed and why:
```
Fix: Changed `tile_size = 128` to `tile_size = 32` in tiling.hpp line 47.
Reason: 128 elements per tile at fp32 (512 bytes) plus workspace exceeded
UB capacity of 192KB. Reducing to 32 keeps total UB usage under the limit.
```

### Step 5: RERUN

Re-run the EXACT command that originally failed. Verify the fix resolves the failure.

```
$ cmake --build build/host
$ ctest --test-dir build/host -R <op_name> --output-on-failure
```

Requirements:
- Use the same command as in Step 1.
- The previously failing test must now PASS (not SKIP, not FAIL with a different error).
- Run the FULL test suite for the operator, not just the previously failing test, to detect regressions:

```
$ ctest --test-dir build/host -R <op_name> --output-on-failure
$ ctest --test-dir build/npu -R <op_name> --output-on-failure  # if NPU available
```

If the fix does NOT resolve the failure, return to Step 2 with the new evidence. Do NOT proceed to Step 6 with a failing test.

If the fix introduces a NEW failure, the fix is incorrect or incomplete. Return to Step 2 and re-isolate.

### Step 6: RECORD

Update the evidence files so future debugging sessions have context.

**Update `BLOCKERS.md`** (if the issue is not yet resolved):
```markdown
## Blocker: <brief title>
- **Component:** kernel / tiling / golden / test / build
- **Symptom:** <what the user/tester sees>
- **Root cause:** <Step 3 explanation>
- **Attempted fixes:** <what was tried>
- **Status:** unresolved | in progress
- **Evidence:** `.npu-harness/runs/<timestamp>/build.log`
```

**Update `STATUS.md`** (if resolved):
```markdown
| debug | resolved | 2026-05-22 | Root cause: <explanation>, fix: <summary> |
```

**Update `DECISIONS.md`** (if the fix involves a design decision):
```markdown
## Decision: <title>
- **Date:** 2026-05-22
- **Context:** <what problem prompted this>
- **Decision:** <what was decided>
- **Rationale:** <why>
- **Alternatives considered:** <what else was considered>
```

## Step-by-Step Procedure (Applied)

### Step 1: Categorize the Failure

Read the failure output. Classify into one of:
- **Build failure:** compiler error, linker error, cmake error.
- **Test failure:** GTest assertion failure.
- **Accuracy failure:** comparison report showing mismatches.
- **Crash/hang:** segmentation fault, abort, infinite loop.
- **Performance anomaly:** unexpected slowdown, no speedup from optimization.
- **Skip anomaly:** test skips when it should run, or passes when it should skip.

### Step 2: Gather Evidence

Collect all relevant artifacts:
- Build log: `.npu-harness/runs/<timestamp>/build.log`.
- Test log: `.npu-harness/runs/<timestamp>/ctest.log`.
- Accuracy report: `.npu-harness/runs/<timestamp>/accuracy.json`.
- Benchmark data: `.npu-harness/runs/<timestamp>/benchmark.json`.
- CMake cache: `build/host/CMakeCache.txt` or `build/npu/CMakeCache.txt`.

### Step 3: Execute the 6-Step Cycle

Follow Steps 1-6 in order: Reproduce, Isolate, Explain, Fix, Rerun, Record.

### Step 4: Escalate If Stuck

If after 3 fix attempts the failure persists:
- Re-examine the root cause explanation. It is likely wrong.
- Read the relevant spec section again. The spec may have an ambiguity.
- Check assumptions about the Ascend C API or hardware behavior.
- Consult the CANN documentation or existing operator examples.
- Write a precise blocker description with all evidence collected so far.

## Anti-Patterns (MUST NOT)

These behaviors defeat the purpose of systematic debugging:

| Anti-pattern | Why it is wrong |
|---|---|
| Fixing symptoms without understanding cause | The real bug remains; the "fix" may hide it or create new bugs. |
| Changing multiple things at once | Cannot determine which change resolved the issue; may introduce side effects. |
| Skipping reproduction | Cannot verify the fix works; may have misunderstood the failure. |
| Changing tolerances to make tests pass | Masks real accuracy bugs. Only adjust tolerances if the spec is provably wrong. |
| Adding `sleep(1)` or similar to "fix" races | The race is still there; timing changes only hide it. |
| Commenting out failing assertions | The bug still exists. Either fix it or document a justified skip. |
| Restarting without recording evidence | Future sessions lose context; the same debugging must be repeated. |
| Blaming the hardware without evidence | "NPU gives wrong results" is not a diagnosis. Show which values, at which indices, with which pattern. |

## Verification Checklist

Before declaring a debugging session complete, verify each item:

- [ ] The failure was reproduced with the exact original command.
- [ ] The exact output was captured to a timestamped log file.
- [ ] The failure was isolated to a specific component (kernel, tiling, golden, test, build config).
- [ ] The root cause was explained in terms of code/spec mismatch, not symptoms.
- [ ] Only ONE cause was fixed per debug iteration.
- [ ] The fix is a minimal, targeted code change with a comment if non-obvious.
- [ ] The previously failing command was re-run and now passes.
- [ ] The FULL test suite for the operator was run and no regressions were introduced.
- [ ] Evidence was recorded: STATUS.md, BLOCKERS.md, and/or DECISIONS.md updated.
- [ ] Run logs are archived in `.npu-harness/runs/<timestamp>/`.
- [ ] If the fix could not be found after 3 attempts, a precise blocker description was written.
