# npu-shipping

## Skill Name

`npu-shipping` -- NPU Operator Shipping

## Trigger Description

Operator development is complete. All phases (intake, spec, plan, implement, build, test, benchmark, verify) have been executed and verified. It is time to produce the final shipping report, update the registry to reflect final status, and hand off the operator for integration or deployment.

## When to Use

- All preceding workflow phases have been completed and verified.
- `verification-before-completion` has been run and returned PASS on all checks.
- The user says "ship <op>" or "finalize <op>" or "wrap up <op>".
- `/npu-ship-op <op_name>` is invoked.
- A milestone requires an operator status summary.
- The operator is being handed off to another team or system.

## When Not to Use

- Any verification check failed -- fix the issues first.
- The operator has not been built or tested -- run the missing phases.
- Accuracy or performance has not been evaluated -- run tests and benchmarks.
- There are unresolved blockers -- resolve them or document them precisely.
- The user just wants a status update -- use `/npu-status` or read OPERATOR_REGISTRY.yaml.
- The operator is in early development -- use the appropriate authoring or implementation skill.

## Required Inputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` -- the completed spec.
2. **INTAKE.yaml** at `.npu-harness/operators/<op_name>/INTAKE.yaml` -- the original intake.
3. **All operator source files** under `ops/<op_name>/` -- kernel, tiling, runner, golden.
4. **Test file** at `tests/test_<op_name>.cpp`.
5. **Build log** at `.npu-harness/runs/<timestamp>/build.log` -- from the most recent successful build.
6. **CTest log** at `.npu-harness/runs/<timestamp>/ctest.log` -- from the most recent successful test run.
7. **Accuracy report** at `.npu-harness/runs/<timestamp>/accuracy.json` -- golden comparison results.
8. **Benchmark data** at `.npu-harness/runs/<timestamp>/benchmark.json` -- if benchmarks were run.
9. **STATUS.md** at `.npu-harness/operators/<op_name>/STATUS.md` -- current status tracking.
10. **BLOCKERS.md** at `.npu-harness/operators/<op_name>/BLOCKERS.md` -- if any blockers exist.
11. **PERF.md** at `.npu-harness/operators/<op_name>/PERF.md` -- if benchmarks were run.
12. **OPERATOR_REGISTRY.yaml** -- the current registry state.

## Required Outputs

1. **FINAL_REPORT.md** at `.npu-harness/operators/<op_name>/FINAL_REPORT.md` -- the shipping report covering all 8 required sections.
2. **Updated OPERATOR_REGISTRY.yaml** -- with final status fields.
3. **Updated STATUS.md** -- with the ship phase marked complete.
4. **Run log** at `.npu-harness/runs/<timestamp>/report.md` -- a copy of the final report in the timestamped run directory.

## Final Report Structure

The shipping report MUST contain these 8 sections in order. Do not omit any section. If a section has no data, write "N/A" or "None" explicitly rather than deleting the section.

### Section 1: Status

```
Status: PASS | PARTIAL | BLOCKED
```

- **PASS:** All required phases complete. Host tests pass. NPU tests pass (or skipped with valid reason). Accuracy within tolerances. Performance targets met (or not required). No unresolved blockers.
- **PARTIAL:** Most phases complete. Some NPU tests skipped because hardware unavailable, but host tests pass. Performance targets not met but gap is documented. Some non-blocking limitations exist.
- **BLOCKED:** Cannot proceed due to a concrete external blocker (missing library, missing hardware, incompatible API). Host tests may still pass.

The status must be one of these three values. Do not invent intermediate statuses.

### Section 2: Commands

Exact commands to reproduce the build and test. The reader must be able to copy-paste these commands and get the same results (assuming the same environment).

```bash
# 1. Configure host build
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo

# 2. Build host
cmake --build build/host

# 3. Run host tests
ctest --test-dir build/host -R <op_name> --output-on-failure

# 4. Configure NPU build (if applicable)
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo

# 5. Build NPU
cmake --build build/npu

# 6. Run NPU tests
ctest --test-dir build/npu -R <op_name> --output-on-failure

# 7. Run benchmarks (if applicable)
python3 scripts/harness.py bench <op_name> --dtype float16 --shape "1,1024,4096"
```

Each command must be an exact string that was successfully executed. Include any environment variables or setup steps needed.

### Section 3: Logs

Key excerpts from `build.log` and `ctest.log`. Do not paste the entire log. Extract:

- Build: last 10 lines (success confirmation or error summary).
- CTest: the summary line ("100% tests passed, 0 tests failed out of N").
- Any warnings worth noting.
- Any skipped tests with their skip reasons.

```text
=== build.log (excerpt) ===
[100/100] Linking CXX executable tests/test_<op_name>
Build succeeded.

=== ctest.log (excerpt) ===
100% tests passed, 0 tests failed out of 12

Test #1: <op_name>_GoldenCorrectness ........   Passed    0.01 sec
Test #2: <op_name>_AllShapes ................   Passed    0.05 sec
...
```

### Section 4: Accuracy

Summary from `accuracy.json`:

```text
Accuracy: PASS | FAIL | PENDING

Configuration       | Elements  | Max Abs Error | Max Rel Error | Mismatches | Status
--------------------|-----------|---------------|---------------|------------|-------
fp32, [1024]        | 1024      | 1.2e-7        | 3.4e-8        | 0          | PASS
fp16, [1024]        | 1024      | 3.1e-4        | 2.8e-4        | 0          | PASS
fp32, [4,16,64]     | 4096      | 8.7e-8        | 5.1e-9        | 0          | PASS
fp16, [4,16,64]     | 4096      | 1.2e-3        | 8.3e-4        | 0          | PASS
```

Include: max absolute error, max relative error, mismatch count, and overall pass/fail for each tested configuration. If any configuration failed, include the first several mismatch locations.

### Section 5: Performance

Summary from `benchmark.json`:

```text
Performance: PASS | FAIL | PENDING | NOT REQUIRED | PENDING_NPU

Dtype   | Shape         | Avg (us) | Min (us) | Max (us) | Target     | Result
--------|---------------|----------|----------|----------|------------|-------
fp16    | 1,1024,4096   | 123.45   | 120.10   | 128.90   | <=1000us   | PASS
fp16    | 2,2048,4096   | 450.12   | 445.30   | 458.70   | <=1000us   | PASS
```

If benchmarks were not run because NPU was unavailable: "PENDING_NPU -- NPU device not available".
If no performance target exists: "NOT REQUIRED -- no performance contract in spec".
If performance target failed: include gap percentage, bottleneck hypothesis, attempted optimizations, and remaining options.

### Section 6: Blockers

Precise blocker descriptions, or "None" if there are no blockers.

```text
Blockers: None
```

OR

```text
Blocker 1: Missing CANN header <acl/acl.h>
  - Component: NPU runner
  - Impact: Cannot compile NPU kernel path
  - Resolution: Install CANN 9.0.0 or set CANN_HOME
  - Evidence: build/npu/CMakeError.log line 42

Blocker 2: No visible Ascend device
  - Component: NPU integration tests
  - Impact: NPU tests skip at runtime
  - Resolution: Ensure device is available and driver is loaded
  - Evidence: detect_ascend_env.py output: "0 devices found"
```

Invalid blocker descriptions: "operator is complex", "Flash Attention is hard", "not sure how to implement", "not enough time". These are not blockers; they are gaps in implementation.

### Section 7: Next Steps

What to do next with this operator:

```text
Next steps:
1. [If PARTIAL due to missing NPU] Deploy to NPU-equipped machine and re-run NPU tests.
2. [If performance target not met] Tune tiling parameters: increase M_TILE from 64 to 128.
3. [If all passing] Operator is ready for integration. Merge to main branch.
4. [If blocked] Resolve block 1 (install CANN headers) before proceeding.
5. Register the operator in the downstream deployment system.
```

### Section 8: Artifact Inventory

All files created or modified for this operator:

```text
Created:
  specs/<op_name>.yaml
  ops/<op_name>/CMakeLists.txt
  ops/<op_name>/kernel/<op_name>_kernel.cpp
  ops/<op_name>/host/<op_name>_tiling.hpp
  ops/<op_name>/host/<op_name>_tiling.cpp
  ops/<op_name>/runner/<op_name>_runner.hpp
  ops/<op_name>/runner/<op_name>_runner.cpp
  ops/<op_name>/golden/<op_name>_golden.hpp
  ops/<op_name>/golden/<op_name>_golden.cpp
  tests/test_<op_name>.cpp
  docs/operators/<op_name>.md
  .npu-harness/operators/<op_name>/INTAKE.yaml
  .npu-harness/operators/<op_name>/STATUS.md
  .npu-harness/operators/<op_name>/FINAL_REPORT.md

Modified:
  OPERATOR_REGISTRY.yaml
  tests/CMakeLists.txt
  ops/CMakeLists.txt (if parent CMake was updated)
```

## Step-by-Step Procedure

### Step 1: Confirm Readiness

Verify that `verification-before-completion` has been run and returned PASS on all applicable checks. If not, stop and run verification first.

Read the STATUS.md to confirm all preceding phases are marked complete:
- [ ] intake: complete
- [ ] spec: complete
- [ ] plan: complete
- [ ] implement: complete
- [ ] build: complete (host, NPU if available)
- [ ] test: complete (host, NPU if available)
- [ ] benchmark: complete (or N/A)

### Step 2: Determine Final Status

Based on all evidence (logs, accuracy, performance, blockers), assign exactly one of: PASS, PARTIAL, BLOCKED.

**PASS requires:**
- Host build: success.
- Host tests: all pass, 0 failures.
- NPU build: success OR skipped with valid reason (no CANN).
- NPU tests: all pass OR all skipped with valid reason.
- Accuracy: all tested configurations within tolerances.
- Performance: all targets met OR not required.
- No unresolved blockers.

**PARTIAL applies when:**
- Some phase is incomplete for a documented, non-blocking reason (e.g., NPU unavailable).
- Performance target not met but gap is precisely documented.
- Some optional dtypes or shapes not yet supported (explicitly deferred).

**BLOCKED applies when:**
- A concrete external dependency is missing.
- Cannot build or test due to environment.
- Critical spec requirement cannot be satisfied.

### Step 3: Collect Evidence

Gather all evidence files:
- Build log excerpt.
- CTest log excerpt.
- Accuracy JSON (parse and summarize).
- Benchmark JSON (parse and summarize).
- Blocker descriptions from BLOCKERS.md.
- File inventory from `git status` or directory listing.

### Step 4: Write the Final Report

Write `.npu-harness/operators/<op_name>/FINAL_REPORT.md` following the 8-section structure above. Use concrete data from evidence files. Do not summarize from memory.

### Step 5: Update OPERATOR_REGISTRY.yaml

Set the final status fields:

```yaml
operators:
  <op_name>:
    spec: "specs/<op_name>.yaml"
    status_file: ".npu-harness/operators/<op_name>/STATUS.md"
    final_report: ".npu-harness/operators/<op_name>/FINAL_REPORT.md"
    category: "<category>"
    generated: true
    built_host: true
    tested_host: true
    built_npu: <true|false|skipped>
    tested_npu: <passed|skipped|failed|pending>
    npu_skip_reason: "<reason if skipped>"
    accuracy: <pass|fail|pending>
    performance: <pass|fail|sanity|pending|pending_npu|not_required>
    final_status: <PASS|PARTIAL|BLOCKED>
```

### Step 6: Update STATUS.md

Mark the ship phase complete:

```markdown
| ship | complete | <date> | Final status: <PASS|PARTIAL|BLOCKED> |
```

### Step 7: Archive to Run Directory

Copy the final report to the timestamped run directory:

```bash
cp .npu-harness/operators/<op_name>/FINAL_REPORT.md \
   .npu-harness/runs/<timestamp>/report.md
```

### Step 8: Present Summary to User

Provide a concise summary:

```
Operator: <op_name>
Final Status: PASS
  Build:   host=PASS, npu=PASS
  Tests:   host=12/12 passed, npu=8/8 passed
  Accuracy: all configurations within tolerances
  Performance: all targets met (best: 12.3% of latency budget)
  Blockers: None
  Next:    Ready for integration. Merge to main branch.
```

## Verification Checklist

Before declaring shipping complete, verify each item:

- [ ] `verification-before-completion` was run and returned PASS on all applicable checks.
- [ ] STATUS.md shows all preceding phases complete.
- [ ] Final status (PASS/PARTIAL/BLOCKED) is correctly assigned per the criteria above.
- [ ] Section 1 (Status): exactly one of PASS, PARTIAL, BLOCKED.
- [ ] Section 2 (Commands): exact, copy-pasteable commands; all were actually executed.
- [ ] Section 3 (Logs): key excerpts from build.log and ctest.log; summary lines included.
- [ ] Section 4 (Accuracy): data from accuracy.json, not from memory; per-configuration table.
- [ ] Section 5 (Performance): data from benchmark.json or explicit status (PENDING_NPU, NOT REQUIRED).
- [ ] Section 6 (Blockers): precise blocker descriptions or "None".
- [ ] Section 7 (Next steps): actionable, specific instructions for each possible path.
- [ ] Section 8 (Artifact inventory): complete list of created and modified files.
- [ ] OPERATOR_REGISTRY.yaml is updated with `final_status` and all evidence-based fields.
- [ ] STATUS.md is updated with ship phase marked complete.
- [ ] Final report is archived to `.npu-harness/runs/<timestamp>/report.md`.
- [ ] No PASS claim if host tests fail.
- [ ] No PASS claim if unresolved blockers exist.
- [ ] No PASS claim if Flash Attention (or any required proof operator) is a stub.
- [ ] No BLOCKED claim without a concrete external dependency identified.
