# /npu-status

Print the current Harness state and a formatted operator status table.

## Usage

```
/npu-status
```

No arguments. Reads all state files and prints a consolidated status view.

## What to Read

Read ALL of these files to construct the status view:

1. `STATE.md` — Harness environment and build status
2. `OPERATOR_REGISTRY.yaml` — Operator inventory (source of truth)
3. `.npu-harness/operators/*/STATUS.md` — Per-operator detailed status
4. `.npu-harness/operators/*/RESULTS.md` — Per-operator verification results
5. `.npu-harness/operators/*/PERF.md` — Per-operator performance results
6. `.npu-harness/operators/*/BLOCKERS.md` — Per-operator blockers
7. `.npu-harness/config.yaml` — Harness configuration
8. `PROJECT.md` — Project overview (for the operator ladder)
9. `ROADMAP.md` — Phase progress

## Output Format

Print the status in clearly sectioned blocks.

### Section 1: Harness Environment

```
============================
 NPU Harness Status
============================

Environment:
  CANN:        <version> (<path>)
  NPU devices: <count> (<device info>)
  cmake:       <version>
  ninja:       <version>
  g++:         <version>
  GTest:       <version>
  python3:     <version>
  Target arch: dav-2201

Build:
  Host: <configured|not configured> / <pass|fail|not built>
  NPU:  <configured|not configured> / <pass|fail|not built> (<skip reason if applicable>)
```

### Section 2: Harness Layer Status

```
Harness Layer:
  CLAUDE.md:              <done|pending>
  AGENTS.md:              <done|pending>
  PROJECT.md:             <done|pending>
  REQUIREMENTS.md:        <done|pending>
  ROADMAP.md:             <done|pending>
  STATE.md:               <done|pending>
  CONTEXT.md:             <done|pending>
  OPERATOR_REGISTRY.yaml: <done|pending>
  Skills:                 <N>/<M> implemented
  Commands:               <N>/<M> implemented
  config.yaml:            <done|pending>
  Templates:              <done|pending>
```

### Section 3: Operator Status Table

This is the primary output. Format as a table:

```
Operator Status:
┌────┬──────────────────────────┬───────────┬───────┬─────────────┬───────────┬──────────┬─────────────┬──────────┐
│ #  │ Operator                 │ Category  │ Gen   │ Host Test   │ NPU Test  │ Accuracy │ Performance │ Shipped  │
├────┼──────────────────────────┼───────────┼───────┼─────────────┼───────────┼──────────┼─────────────┼──────────┤
│ 1  │ elementwise_add          │ elem      │ done  │ pass 8/8    │ skip(C)   │ pass     │ sanity      │ no       │
│ 2  │ axpy_scalar_affine       │ elem      │ done  │ pass 6/6    │ skip(C)   │ pass     │ not_req     │ no       │
│ 3  │ broadcast_binary         │ elem      │ done  │ pass 10/10  │ skip(C)   │ pass     │ not_req     │ no       │
│ 4  │ activation_gelu_silu     │ elem      │ stub  │ -           │ -         │ -        │ -           │ no       │
│ 5  │ reduce_sum_lastdim       │ reduction │ stub  │ -           │ -         │ -        │ -           │ no       │
│ 6  │ transpose_2d_4d          │ layout    │ -     │ -           │ -         │ -        │ -           │ no       │
│ 7  │ layer_norm               │ norm      │ -     │ -           │ -         │ -        │ -           │ no       │
│ 8  │ matmul_bias              │ matmul    │ -     │ -           │ -         │ -        │ -           │ no       │
│ 9  │ rms_norm_rope_fused      │ fusion    │ -     │ -           │ -         │ -        │ -           │ no       │
│ 10 │ flash_attention_fwd_fused│ attention │ -     │ -           │ -         │ -        │ -           │ no       │
└────┴──────────────────────────┴───────────┴───────┴─────────────┴───────────┴──────────┴─────────────┴──────────┘

Legend:
  Gen:         done = scaffold generated, stub = scaffold but empty, - = not started
  Host Test:   pass N/M = N passed out of M, fail = tests fail, - = no tests
  NPU Test:    pass N/M, fail N/M, skip(C) = CANN not detected, skip(D) = no device, - = no tests
  Accuracy:    pass, fail, pending, unknown
  Performance: pass, gap = below target, sanity = benchmarked no target, not_req = not required, pending_npu = waiting for NPU
  Shipped:     yes = /npu-ship-op completed, no = not shipped
```

### Section 4: Active Blockers

```
Active Blockers:
  <op_name>: <blocker description>
  <op_name>: <blocker description>

  No active blockers.
```

If no blockers exist, print "No active blockers."

### Section 5: Recent Runs

List the most recent timestamped runs:

```
Recent Runs:
  .npu-harness/runs/20260522_143021/ — map-codebase
  .npu-harness/runs/20260522_150512/ — verify elementwise_add
  .npu-harness/runs/20260522_153045/ — bench matmul_bias
```

### Section 6: Next Actions

Based on the current state, suggest the next logical action:

```
Next Actions:
  1. /npu-new-op <<<...>>> — Add a new operator
  2. /npu-execute-op activation_gelu_silu — Implement the next pending operator
  3. /npu-doctor — Diagnose environment issues
```

## Data Extraction from Files

### From OPERATOR_REGISTRY.yaml

For each operator, extract:
- `spec` path
- `category`
- `generated`
- `built_host`, `tested_host`
- `built_npu`, `tested_npu`, `npu_skip_reason`
- `accuracy`, `performance`
- `shipped` (if present)

### From STATUS.md

For per-operator granularity, read each `.npu-harness/operators/<op_name>/STATUS.md` and extract the phase status table:

```
| Phase | Status | Notes |
```

This gives more detail than the registry for in-progress operators.

### From RESULTS.md

Extract test counts (total/passed/failed/skipped) for Host and NPU categories.

### From PERF.md

Extract the "Overall" performance verdict and whether targets were met.

### From BLOCKERS.md

List each blocker with the operator name.

## Rules

1. OPERATOR_REGISTRY.yaml is the source of truth. If a STATUS.md disagrees with the registry, flag the discrepancy.
2. Use "-" for "not started / no data" rather than guessing.
3. NPU skip reasons must be precise: "CANN not detected" vs "No device" vs "Not yet implemented".
4. If no operators exist in the registry, show an empty table with a note: "No operators registered. Use /npu-new-op to create one."
5. If OPERATOR_REGISTRY.yaml does not exist, print a clear message: "OPERATOR_REGISTRY.yaml not found. Run /npu-map-codebase first."
6. The status command is read-only. It must not modify any files.
7. If a file cannot be read, note it but continue. A missing file is not an error.

## Output Variants

### No Operators Yet

```
============================
 NPU Harness Status
============================

Environment: <env info>

No operators registered yet.

Next action: /npu-new-op <<<describe your first operator>>>
```

### All Operators Shipped

```
============================
 NPU Harness Status
============================

<operator table with all rows marked shipped: yes>

All 10 operators shipped!
```

### Mixed State (typical)

Print the full table with a mix of statuses, followed by next actions that prioritize unshipped operators.
