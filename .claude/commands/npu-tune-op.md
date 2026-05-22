# /npu-tune-op

Benchmark and tune the operator's performance until the performance contract passes or a precise performance gap report is produced.

## Usage

```
/npu-tune-op <op_name>
```

## Arguments

- `op_name` — The operator name to benchmark and tune.

## Preconditions

- Operator implementation is complete (`/npu-execute-op` succeeded)
- Verification passed (`/npu-verify-op` succeeded or host tests at minimum pass)
- SPEC.yaml has a `performance` section
- NPU build exists (`build/npu/`) if NPU is available

If `performance.required: false` in the spec, this command still runs sanity benchmarks but does not enforce a target.

If NPU is not available, tuning is limited to host-only benchmarks.

## What to Read First

1. `.npu-harness/operators/<op_name>/SPEC.yaml` — The performance contract (targets section)
2. `.npu-harness/operators/<op_name>/PLAN.md` — The performance plan and tuning strategy
3. `.npu-harness/operators/<op_name>/RESULTS.md` — Verification results for baseline
4. `.npu-harness/config.yaml` — Benchmark defaults (warmup, iterations)

## Workflow

### Step 1: Check Tuning Preconditions

Determine what level of tuning is possible:

```bash
python3 scripts/detect_ascend_env.py
```

**Case A: NPU available, performance.required: true**
- Run full benchmarking on NPU
- Compare to targets
- Tune if needed

**Case B: NPU available, performance.required: false**
- Run sanity benchmarks on NPU
- Record measurements but no target enforcement
- Skip tuning loop

**Case C: NPU not available**
- Run host-only benchmarks if applicable
- Record in PERF.md: `"NPU not available — performance measurement skipped"`
- Skip to Step 6 (produce PERF.md with skip notice)

### Step 2: Baseline Benchmark

Run the baseline benchmark:

```bash
python3 scripts/harness.py bench <op_name>
```

If the harness CLI doesn't yet support `bench`, build and run the benchmark target directly:

```bash
cmake --build build/npu --target bench_<op_name>
./build/npu/ops/<op_name>/bench/bench_<op_name>
```

The benchmark output must include for each measured configuration:

```
operator=<op_name>
dtype=<dtype>
shape=<shape>
device=<npu|cpu>
arch=dav-2201
warmup=<N>
iterations=<N>
avg_us=<value>
min_us=<value>
max_us=<value>
<optional: bandwidth_gb_s or flops>
```

### Step 3: Compare to Targets

For each target in SPEC.yaml `performance.targets`, compare the measured value:

```
Target: shape=B=1,S=1024,H=4096, dtype=float16, metric=latency_us, target=1000, comparison="<="
Measured: 1234 us
Gap: +234 us (+23.4%) — DOES NOT MEET TARGET
```

If ALL targets pass:
- Tuning is complete. Skip to Step 5.
- Record: `"All performance targets met. No tuning required."`

If ANY target fails:
- Proceed to Step 4 (tuning loop).

### Step 4: Tuning Loop

The tuning loop iterates up to `performance.tuning_budget.max_iterations` times (default 5).

For each iteration:

#### 4.1: Analyze Bottleneck

Inspect the benchmark results and identify the likely bottleneck:

| Symptom | Likely Bottleneck |
|---------|-------------------|
| Time scales linearly with elements but bandwidth is far below peak | Memory bandwidth bound |
| Time scales super-linearly | Compute bound or cache thrashing |
| Large gap between min and max | Variability (other processes, thermal throttling) |
| Same time for small and large shapes | Launch overhead dominant |
| Tail elements disproportionately slow | Tail handling inefficiency |

Use the `performance-benchmarking` skill for guidance.

#### 4.2: Formulate Hypothesis

State the bottleneck hypothesis explicitly:

```
Hypothesis: The kernel is memory-bandwidth bound. Current tiling uses tile_size=128.
Increasing tile_size to 256 should improve bandwidth utilization.
```

#### 4.3: Apply Change

Make ONE change per iteration:
- Adjust tile sizes (increase or decrease)
- Adjust block dimensions
- Adjust work distribution across cores
- Add or modify vectorized memory operations
- Modify data layout or access pattern
- Add double buffering if applicable

Do NOT make multiple changes in one iteration. You cannot attribute the effect.

Record the change in PERF.md:
```markdown
### Iteration <N>
- **Change:** <what was changed>
- **Hypothesis:** <why this should help>
```

#### 4.4: Re-benchmark

```bash
# Rebuild with the change
cmake --build build/npu

# Re-run benchmark
python3 scripts/harness.py bench <op_name>
```

#### 4.5: Compare

Compare before and after:

```markdown
- **Before:** <avg_us> us
- **After:** <avg_us> us
- **Delta:** <+-X%>
- **Target:** <target_us> us (comparison: <=)
- **Target met:** <YES|NO>
```

#### 4.6: Decide

- If target passes: exit tuning loop (SUCCESS).
- If iteration < max_iterations AND improvement was meaningful: continue to next iteration.
- If iteration < max_iterations AND improvement was negligible (< 2%): reconsider hypothesis.
- If iteration >= max_iterations: exit with GAP REPORT.

### Step 5: Verify Performance (if targets met)

```bash
python3 scripts/harness.py verify-performance <op_name>
```

This is a final check that the performance contract is satisfied.

### Step 6: Produce PERF.md

Write `.npu-harness/operators/<op_name>/PERF.md`:

```markdown
# Performance Report: <op_name>

**Date:** <today>
**Device:** <npu|cpu>
**Arch:** dav-2201

## Performance Contract

<Copy from SPEC.yaml performance section>

## Baseline

| Shape | Dtype | Metric | Avg | Min | Max | Target | Comparison | Status |
|-------|-------|--------|-----|-----|-----|--------|------------|--------|
| <shape> | <dtype> | <metric> | <v> | <v> | <v> | <target> | <= | <PASS|FAIL> |

## Tuning Iterations

### Iteration 1
- **Change:** <description>
- **Hypothesis:** <rationale>
- **Before:** <avg> us
- **After:** <avg> us
- **Delta:** <+-X%>
- **Target met:** <YES|NO>

### Iteration 2
...

## Final Results

| Shape | Dtype | Metric | Measured | Target | Status |
|-------|-------|--------|----------|--------|--------|
| <shape> | <dtype> | <metric> | <v> | <target> | <PASS|FAIL|GAP> |

## Tuning Summary

- **Iterations used:** <N> / <max>
- **Targets met:** <K> / <M>
- **Overall:** <PASS|GAP REPORTED|SKIPPED>

<If GAP REPORTED:>

## Performance Gap Report

| Target | Measured | Gap | Gap % |
|--------|----------|-----|-------|
| <target> | <v> | <v> | <X%> |

### Likely Bottleneck
<description>

### Attempted Optimizations
1. <change 1> — <result>
2. <change 2> — <result>

### Remaining Options
1. <option 1> — <feasibility>
2. <option 2> — <feasibility>

### Blocker
<If there is a concrete external blocker preventing further optimization.>
```

### Step 7: Produce Benchmark JSON

Create `.npu-harness/runs/<timestamp>/benchmark.json`:

```json
{
  "operator": "<op_name>",
  "timestamp": "<ISO 8601>",
  "device": "<npu|cpu>",
  "arch": "dav-2201",
  "results": [
    {
      "shape": "<shape>",
      "dtype": "<dtype>",
      "metric": "latency_us",
      "target": <value>,
      "comparison": "<=",
      "avg": <value>,
      "min": <value>,
      "max": <value>,
      "iterations": <N>,
      "warmup": <N>,
      "passed": true
    }
  ],
  "tuning_iterations": <N>,
  "targets_met": <K>,
  "targets_total": <M>,
  "overall": "<PASS|GAP_REPORTED|SKIPPED>"
}
```

### Step 8: Update STATUS.md

```markdown
| benchmark | done | <PASS|GAP|SKIP> |
| tune | done | <iterations used> |
```

## Tuning Stop Conditions

Stop the tuning loop when ANY of these is true:

1. **All targets pass.** Success. Record and ship.

2. **Tuning budget exhausted.** Max iterations reached and targets not met. Write a precise performance gap report. This is NOT a failure of the Harness; it is a documented gap.

3. **Concrete external blocker exists.** For example:
   - Missing CANN profiling tools prevent bottleneck identification
   - Required Ascend C API not available in current CANN version
   - Hardware limitation (e.g., memory bandwidth fundamentally insufficient for target)
   - NPU device not accessible

   Document the blocker in BLOCKERS.md:
   ```markdown
   ## Performance Blocker
   - **Blocker:** <specific missing capability>
   - **Impact:** <what target cannot be met>
   - **Evidence:** <measurement data supporting the claim>
   - **Mitigation:** <workaround if any>
   ```

4. **Performance is not required.** (`performance.required: false`). Run sanity benchmarks, record results, but do not enter tuning loop.

Invalid stop conditions:
- "Tuning is hard"
- "The gap is large"
- "We tried a few things"
- Giving up without writing a gap report

## Rules

1. Change ONE thing per tuning iteration. Multiple simultaneous changes prevent root cause analysis.
2. Measure before and after every change. No subjective "feels faster."
3. Record every iteration in PERF.md. No silent tuning.
4. If NPU is unavailable, record "NPU not available" in PERF.md and skip tuning. Do not pretend to tune on host.
5. A performance gap report is a valid and useful output. It is not a failure.
6. Never claim a performance target is met without benchmark evidence.
7. Warmup iterations are mandatory to avoid cold-start artifacts.
8. Multiple measurement iterations are mandatory to detect variance.

## Output

After tuning, print a summary:

```
Tuning complete for <op_name>.

Targets: <K>/<M> met
Iterations used: <N> / <max>
Overall: <PASS|GAP REPORTED|SKIPPED>

Perf report:  .npu-harness/operators/<op_name>/PERF.md
Benchmark:    .npu-harness/runs/<timestamp>/benchmark.json

Next command: /npu-ship-op <op_name>
```
