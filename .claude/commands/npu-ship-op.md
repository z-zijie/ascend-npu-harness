# /npu-ship-op

Finalize the operator, update the operator registry, and produce a shipping report. This is the last command before an operator is considered complete.

## Usage

```
/npu-ship-op <op_name>
```

## Arguments

- `op_name` — The operator name to finalize.

## Preconditions

All of these must be true before shipping. If any are missing or failing, do not ship — fix the issue first.

### Mandatory Preconditions

1. **SPEC.yaml exists** — `.npu-harness/operators/<op_name>/SPEC.yaml`
2. **Documentation exists** — `docs/operators/<op_name>.md`
3. **CPU golden exists** — `ops/<op_name>/golden/<op_name>_golden.cpp` (implemented, not stubbed)
4. **Host tests pass** — Verified by `/npu-verify-op`
5. **Accuracy report exists** — `.npu-harness/operators/<op_name>/RESULTS.md` (or equivalent accuracy data)

### Conditional Preconditions

6. **If performance.required: true** — Benchmark report must exist (`.npu-harness/operators/<op_name>/PERF.md`) and either targets pass OR a gap report exists.
7. **If NPU available** — NPU tests must pass or produce documented skips with reasons.
8. **If implementation is incomplete** — Blocker report must exist (`.npu-harness/operators/<op_name>/BLOCKERS.md`).

## What to Read Before Shipping

1. `.npu-harness/operators/<op_name>/SPEC.yaml` — The full contract
2. `.npu-harness/operators/<op_name>/STATUS.md` — Current status
3. `.npu-harness/operators/<op_name>/RESULTS.md` — Verification results
4. `.npu-harness/operators/<op_name>/PERF.md` — Performance results (if applicable)
5. `.npu-harness/operators/<op_name>/BLOCKERS.md` — Any unresolved blockers
6. `.npu-harness/operators/<op_name>/PLAN.md` — What was planned vs. what was done
7. `.npu-harness/operators/<op_name>/DECISIONS.md` — Design decisions

## Shipping Checklist

Run through every item. Mark each as PASS, SKIP (with reason), or FAIL (with details).

### 1. Spec Completeness

```markdown
- [ ] SPEC.yaml has name, category, target_arch
- [ ] All inputs have complete dtype and shape specifications
- [ ] All outputs have complete dtype and shape specifications
- [ ] math.definition is clear and implementable
- [ ] Accuracy tolerances defined for all supported dtypes
- [ ] Performance contract present (if required)
- [ ] Test shapes specified
```

### 2. Documentation Completeness

```markdown
- [ ] docs/operators/<op_name>.md exists
- [ ] Mathematical definition present
- [ ] Input/output shapes documented
- [ ] Dtype support documented
- [ ] Tiling strategy documented (if applicable)
- [ ] Direct launch notes present (if NPU supported)
- [ ] CPU golden description present
- [ ] Test coverage summary present
- [ ] Expected tolerances documented
- [ ] Known limitations documented
```

### 3. Implementation Completeness

```markdown
- [ ] ops/<op_name>/golden/<op_name>_golden.cpp — CPU golden implemented
- [ ] ops/<op_name>/host/<op_name>_tiling.cpp — Shape validation implemented
- [ ] ops/<op_name>/host/<op_name>_tiling.cpp — Tiling implemented
- [ ] ops/<op_name>/kernel/<op_name>_kernel.cpp — NPU kernel implemented (or stubbed with blocker)
- [ ] ops/<op_name>/runner/<op_name>_runner.cpp — Runner implemented (or stubbed with blocker)
- [ ] ops/<op_name>/CMakeLists.txt — CMake integration complete
- [ ] ops/<op_name>/bench/bench_<op_name>.cpp — Benchmark target exists
- [ ] ops/<op_name>/README.md — Operator README exists
```

### 4. Test Completeness

```markdown
- [ ] tests/test_<op_name>.cpp exists
- [ ] Shape coverage: small, medium, large, edge cases
- [ ] Dtype coverage: all supported dtypes
- [ ] Invalid input rejection tests
- [ ] Golden comparison tests
- [ ] NPU vs golden comparison tests (or skip with reason)
- [ ] CTest labels applied (<op_name>, <category>, host, golden, npu)
- [ ] All host tests pass
- [ ] All NPU tests pass (or skip with clear reason)
```

### 5. Accuracy Completeness

```markdown
- [ ] RESULTS.md exists with accuracy data
- [ ] Accuracy within tolerance for all tested dtypes
- [ ] accuracy.json exists in .npu-harness/runs/<timestamp>/
- [ ] Mismatch details recorded (if any mismatches within tolerance)
```

### 6. Performance Completeness

```markdown
- [ ] If performance.required: true — PERF.md exists
- [ ] If performance.required: true — Targets met or gap report written
- [ ] If performance.required: false — Sanity benchmarks recorded (optional)
- [ ] benchmark.json exists (if benchmarks were run)
```

### 7. Blocker Completeness

```markdown
- [ ] If any precondition is missing — BLOCKERS.md exists
- [ ] Each blocker is specific (missing header, library, API, device)
- [ ] Blockers are actionable (what would resolve them)
- [ ] No generic blockers ("too complex", "not enough time")
```

### 8. Registry Up-to-Date

```markdown
- [ ] OPERATOR_REGISTRY.yaml entry is current
- [ ] spec field points to correct spec file
- [ ] status_file field points to STATUS.md
- [ ] generated, built_host, tested_host fields are accurate
- [ ] built_npu, tested_npu fields are accurate
- [ ] npu_skip_reason is filled if NPU was skipped
- [ ] accuracy field reflects actual results
- [ ] performance field reflects actual results
```

## Actions

### A. Update OPERATOR_REGISTRY.yaml

Set all fields to their final values based on the shipping checklist:

```yaml
operators:
  <op_name>:
    spec: specs/<op_name>.yaml
    status_file: .npu-harness/operators/<op_name>/STATUS.md
    category: <category>
    generated: true
    built_host: true
    tested_host: true
    built_npu: <true|false|skipped>
    tested_npu: <true|false|skipped>
    npu_skip_reason: "<reason if skipped, else null>"
    accuracy: <pass|fail>
    performance: <pass|fail|gap_reported|sanity|pending_npu|not_required>
    shipped: true
    ship_date: <today>
```

### B. Update STATUS.md

```markdown
| ship | done | <date> |
```

### C. Create Shipping Report

Write `.npu-harness/runs/<timestamp>/report.md`:

```markdown
# Shipping Report: <op_name>

**Date:** <today>
**Status:** <SHIPPED|SHIPPED_WITH_GAPS|BLOCKED>

## Checklist Results

### Spec
<Pass/Fail/Skip per item>

### Documentation
<Pass/Fail/Skip per item>

### Implementation
<Pass/Fail/Skip per item>

### Tests
<Pass/Fail/Skip per item>

### Accuracy
<Pass/Fail/Skip per item>

### Performance
<Pass/Fail/Skip per item>

### Blockers
<List or "None">

## Summary

**Operator <op_name> is <SHIPPED|SHIPPED_WITH_GAPS|BLOCKED>.**

<If SHIPPED:>
All mandatory preconditions are satisfied. The operator is ready for use.

<If SHIPPED_WITH_GAPS:>
The operator passes all correctness checks but has documented gaps:
- <gap 1>
- <gap 2>

<If BLOCKED:>
The operator cannot be shipped due to:
- <blocker 1>
- <blocker 2>

## Files

- Spec: `.npu-harness/operators/<op_name>/SPEC.yaml`
- Plan: `.npu-harness/operators/<op_name>/PLAN.md`
- Decisions: `.npu-harness/operators/<op_name>/DECISIONS.md`
- Status: `.npu-harness/operators/<op_name>/STATUS.md`
- Results: `.npu-harness/operators/<op_name>/RESULTS.md`
- Performance: `.npu-harness/operators/<op_name>/PERF.md`
- Blockers: `.npu-harness/operators/<op_name>/BLOCKERS.md`
- Registry: `OPERATOR_REGISTRY.yaml`
```

## Shipping Decisions

Based on the checklist results, determine the shipping status:

### SHIPPED

All mandatory preconditions are met:
- Spec, docs, and CPU golden exist
- Host tests pass
- Accuracy report exists and passes
- If performance required: targets pass or gap report exists
- If NPU available: NPU tests pass (or have documented skips)
- No unresolved blockers

### SHIPPED_WITH_GAPS

All correctness preconditions are met, but there are documented gaps:
- Performance targets not met but gap report exists
- NPU not available but kernel code is written
- Some optional features not implemented
- Edge cases documented as known limitations

The gaps MUST be documented. If gaps are not documented, this is not valid.

### BLOCKED

One or more mandatory preconditions cannot be met due to concrete external blockers:
- CANN not available and NPU is required for this operator
- Missing API in current CANN version
- Hardware limitation
- Missing external dependency

The blockers MUST be specific and actionable.

NOT valid shipping statuses:
- "Almost done"
- "Mostly works"
- "Would be shipped if we had more time"
- SHIPPED with failing host tests
- SHIPPED with missing CPU golden

## Rules

1. Do not ship if host tests fail. This is a hard block.
2. Do not ship if the CPU golden is missing or stubbed. This is a hard block.
3. Do not ship if the spec is missing critical fields (no shape, no dtype, no accuracy tolerance).
4. Do not claim SHIPPED if NPU was never tested AND NPU was available. Document the skip or fix the issue.
5. A documented gap is acceptable. An undocumented gap is not.
6. Every shipped operator must have an entry in OPERATOR_REGISTRY.yaml.
7. The shipping report must be honest. It is better to ship with documented gaps than to claim perfection falsely.

## Output

After shipping, print a final summary:

```
Shipping complete for <op_name>.
Status: <SHIPPED|SHIPPED_WITH_GAPS|BLOCKED>

Checklist: <N>/<M> items passed
Blockers: <count or "None">
Gaps: <count or "None">

Registry updated: OPERATOR_REGISTRY.yaml
Shipping report: .npu-harness/runs/<timestamp>/report.md

The operator is <ready for use | ready with documented gaps | blocked — see report>.
```
