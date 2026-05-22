# performance-benchmarking

## Skill Name

`performance-benchmarking` -- Performance Benchmarking

## Trigger Description

The user requests benchmarking an operator's performance. This happens when a performance target exists in the SPEC.yaml, when sanity benchmarks are needed, after a tuning iteration, or when comparing before/after performance of a change.

## When to Use

- SPEC.yaml has `performance.required: true` and a performance target is defined.
- The user says "benchmark <op>" or "measure performance of <op>".
- `/npu-tune-op <op_name>` is invoked and benchmarks are needed for each tuning iteration.
- A code change was made and its performance impact must be quantified.
- Regression detection: comparing current performance against a previous benchmark run.
- Sanity benchmarks are desired even when `performance.required: false`.
- A performance gap report needs data.

## When Not to Use

- The operator has no performance target and the user does not request benchmarks.
- The operator is still in early implementation and is known to be incorrect -- fix correctness first, then benchmark.
- The task is to run correctness tests -- use `gtest-verification`.
- The task is to analyze a crash or hang -- use `systematic-debugging`.
- CANN/NPU is unavailable and the user expects NPU numbers -- the benchmark will skip, but host benchmarks (CPU golden timing) can still run if useful.

## Required Inputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` -- provides `benchmark.warmup`, `benchmark.iterations`, and `performance.targets`.
2. **Built NPU target** (if benchmarking NPU) -- the kernel and runner must be compiled.
3. **Previous benchmark.json** at `.npu-harness/runs/<timestamp>/benchmark.json` (for regression detection, if available).
4. **Benchmark executable** at `ops/<op_name>/bench/bench_<op_name>.cpp` or via `harness.py bench <op_name>`.

## Required Outputs

1. **Benchmark data** written to `.npu-harness/runs/<timestamp>/benchmark.json` in harness schema.
2. **Target comparison** -- pass/fail with measured gap.
3. **Regression report** -- if previous benchmarks exist, compare and flag regressions.
4. **Updated PERF.md** at `.npu-harness/operators/<op_name>/PERF.md` with benchmark results and analysis.
5. **Updated STATUS.md** -- reflect benchmark results.

## Benchmark JSON Schema

```json
{
  "schema_version": 1,
  "operator": "<op_name>",
  "target_arch": "dav-2201",
  "timestamp": "<ISO 8601>",
  "config": {
    "warmup_iterations": 10,
    "measurement_iterations": 100,
    "dtype": "float16",
    "shape": [1, 1024, 4096],
    "device": "npu",
    "device_id": 0
  },
  "results": {
    "avg_time_us": 123.45,
    "min_time_us": 120.10,
    "max_time_us": 128.90,
    "stddev_time_us": 2.34,
    "median_time_us": 123.00,
    "p99_time_us": 127.50
  },
  "derived": {
    "bandwidth_gb_s": 98.7,
    "tflops": 2.1,
    "elements_per_second": 1.2e9
  },
  "target": {
    "metric": "latency_us",
    "target": 1000.0,
    "comparison": "<=",
    "measured": 123.45,
    "passed": true,
    "gap_pct": -87.7
  },
  "regression": {
    "baseline_timestamp": "<ISO 8601 or null>",
    "baseline_avg_us": 115.0,
    "delta_pct": 7.3,
    "regression_detected": true,
    "threshold_pct": 5.0
  },
  "notes": ""
}
```

## Mandatory Requirements

### Requirement 1: Warmup Iterations

Always run warmup iterations before measurement. Default: 10. The warmup loop must exercise the exact same code path as measurement but discard the results. This eliminates cold-start effects (first kernel launch overhead, cache warming, memory allocation overhead).

```cpp
// Warmup
for (int i = 0; i < warmup_iters; ++i) {
    runner.Run(input, output);  // discard results
}
device_context.Synchronize();
```

### Requirement 2: Measurement Iterations

Run enough iterations for statistical significance. Default: 100. For very slow operators (>1s per iteration), reduce to 10 with a note in `config.notes`. For very fast operators (<1us per iteration), increase to 1000 or use a batch-repeat loop to amortize launch overhead.

### Requirement 3: Average/Min/Max Time Reported

For each benchmark configuration, report:
- Average (arithmetic mean) time.
- Minimum time.
- Maximum time.
- Standard deviation.
- Median time (P50).
- P99 time (optional but recommended for latency-sensitive operators).

### Requirement 4: Bandwidth or FLOPs Where Meaningful

Compute derived metrics when they are well-defined:

- **Bandwidth (GB/s):** `(input_bytes + output_bytes) / avg_time_seconds / 1e9`
- **FLOPs (TFLOPS):** `total_floating_point_operations / avg_time_seconds / 1e12`
- **Elements/second:** `total_elements / avg_time_seconds`

For elementwise operations, bandwidth is the most meaningful metric. For matmul, FLOPs is more meaningful. For layout transforms, bandwidth is appropriate. For attention, both can be reported.

If the operation count is not easily computed or the metric is not meaningful, omit it rather than fabricating a number.

### Requirement 5: Target Comparison

If a performance target exists in SPEC.yaml, compare the measured value to the target:

```
target:  1000 us
measured: 123 us
comparison: <=
result: PASS (12.3% of target, 87.7% headroom)
```

If target comparison fails:

```
target:  100 us
measured: 145 us
comparison: <=
result: FAIL (45% over target, gap = 45 us)
```

The comparison must use the operator from the spec (`<=`, `>=`, `==`). Report both the absolute gap and the percentage gap.

### Requirement 6: Regression Detection

If a previous benchmark exists for the same (op, dtype, shape, device) configuration, compare:

```
Baseline: 115.0 us (from run 2026-05-21T10:00:00Z)
Current:  123.5 us
Delta:    +7.4%
Threshold: 5.0%
Result:   REGRESSION DETECTED
```

The regression threshold is 5% by default. Document the threshold used. If no baseline exists, set `regression.baseline_timestamp: null` and `regression.regression_detected: false`.

### Requirement 7: Output Format Matches Harness Schema

The benchmark output MUST be valid JSON conforming to the harness benchmark schema shown above. The JSON must be written to `.npu-harness/runs/<timestamp>/benchmark.json`.

### Requirement 8: Graceful Skip When Device Unavailable

When NPU is unavailable, benchmarks must skip with a clear message:

```
[N/A] NPU not available; skipping benchmark for <op_name>.
```

The benchmark executable must exit with code 0 (not error) when skipping. The benchmark JSON should still be written with `"device": "npu"` and `"skipped": true` with a `"skip_reason"` field.

## Step-by-Step Procedure

### Step 1: Check the Performance Requirements

Read `specs/<op_name>.yaml` and extract:
- `performance.required` (true/false).
- `performance.targets` list (if any).
- `benchmark.warmup` (default 10).
- `benchmark.iterations` (default 100).

If `performance.required: false` and the user has not explicitly asked for benchmarks, and there are no targets, ask whether sanity benchmarks are desired.

### Step 2: Build the Benchmark Target

Ensure the operator benchmark is built:

```bash
cmake --build build/npu --target bench_<op_name>
```

If NPU is unavailable and only host benchmarks are meaningful:

```bash
cmake --build build/host --target bench_<op_name>
```

### Step 3: Create the Run Directory

```bash
mkdir -p .npu-harness/runs/$(date +%Y%m%dT%H%M%S)
```

Record the timestamp for the JSON output.

### Step 4: Run the Benchmark

Invoke the benchmark executable with appropriate arguments:

```bash
# Via harness CLI (preferred)
python3 scripts/harness.py bench <op_name> \
    --dtype float16 \
    --shape "1,1024,4096" \
    --warmup 10 \
    --iterations 100 \
    --output .npu-harness/runs/<timestamp>/benchmark.json

# Or directly
./build/npu/ops/<op_name>/bench_<op_name> \
    --dtype float16 \
    --shape 1 1024 4096 \
    --warmup 10 \
    --iterations 100 \
    --json-output .npu-harness/runs/<timestamp>/benchmark.json
```

### Step 5: Collect and Validate Results

Check that the benchmark JSON contains all required fields. Verify:
- avg_time_us > 0.
- min_time_us <= avg_time_us <= max_time_us.
- Standard deviation is non-negative.
- Derived metrics are computed correctly.

### Step 6: Compare Against Performance Target

For each target in `performance.targets`, find the matching benchmark result and compute:

```
passed = (measured COMPARISON target)
gap_pct = (measured - target) / target * 100  // positive means over target
```

Write the comparison into the JSON. If no target exists, set `target: null`.

### Step 7: Check for Regressions

Search `.npu-harness/runs/*/benchmark.json` for the most recent previous run of the same (op, dtype, shape, device). If found:

- Read baseline `avg_time_us`.
- Compute `delta_pct = (current_avg - baseline_avg) / baseline_avg * 100`.
- If `|delta_pct| > threshold` and `delta_pct > 0`, flag as regression.

Write regression data into the JSON.

### Step 8: Write PERF.md

Update `.npu-harness/operators/<op_name>/PERF.md`:

```markdown
# Performance: <op_name>

## Benchmark Summary

| Dtype | Shape | Device | Avg (us) | Min (us) | Max (us) | Bandwidth (GB/s) | Target | Result |
|-------|-------|--------|----------|----------|----------|------------------|--------|--------|
| fp16  | 1,1024,4096 | npu | 123.45 | 120.10 | 128.90 | 98.7 | <=1000us | PASS |
| fp16  | 2,2048,4096 | npu | 450.12 | 445.30 | 458.70 | 110.2 | <=1000us | PASS |

## Performance Analysis

- Best case: shape (B=1,S=1024,H=4096), achieves 12.3% of latency target.
- Bottleneck hypothesis: GM bandwidth, not compute. Bandwidth utilization ~8% of peak.
- Tuning opportunities: increase tile sizes to improve GM utilization; double-buffer K loop.
```

### Step 9: Update STATUS.md

Add benchmark status:

```
| benchmark | complete | 2026-05-22 | All targets passed, no regressions |
```

## Verification Checklist

Before declaring benchmarking complete, verify each item:

- [ ] Warmup iterations were executed (default 10).
- [ ] Measurement iterations were executed (default 100).
- [ ] Average, min, max, stddev, and median times are reported.
- [ ] Bandwidth or FLOPs are computed where meaningful (non-trivial ops).
- [ ] Each performance target has a pass/fail comparison.
- [ ] Gap percentage is computed for targets that failed or have headroom.
- [ ] Regression detection was run if previous benchmarks exist.
- [ ] Regression threshold is documented (default 5%).
- [ ] Benchmark JSON conforms to the harness schema.
- [ ] Benchmark JSON is written to `.npu-harness/runs/<timestamp>/benchmark.json`.
- [ ] NPU benchmarks skip gracefully when device unavailable.
- [ ] PERF.md is updated with results and analysis.
- [ ] STATUS.md is updated.
- [ ] All benchmarking commands are logged in the harness run log.
- [ ] No benchmark passes by accident (e.g., empty data, 0 iterations, wrong comparison direction).
- [ ] Timestamp is recorded in ISO 8601 format.
