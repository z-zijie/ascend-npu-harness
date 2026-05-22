# npu-spec-authoring

## Skill Name

`npu-spec-authoring` -- Operator SPEC Authoring

## Trigger Description

The user requests writing or updating an operator SPEC file. This typically follows `npu-operator-intake` and `/npu-discuss-op`. It may also be invoked when an existing SPEC needs revision due to design changes, review feedback, or newly discovered constraints.

## When to Use

- An INTAKE.yaml exists at `.npu-harness/operators/<op_name>/INTAKE.yaml` and is ready for SPEC generation.
- The user says "write the spec for <op>" or "create SPEC.yaml for <op>".
- An existing SPEC.yaml needs amendment (new dtype support, revised tolerances, added mask support).
- `/npu-discuss-op` has resolved all open questions and the spec can be finalized.
- A code review reveals that the implementation does not match the spec, and the spec is the authoritative source that needs updating.

## When Not to Use

- No INTAKE.yaml exists -- run `npu-operator-intake` first.
- The user wants to implement code, not write a spec -- use `ascend-c-kernel-implementation` or `cpu-golden-authoring`.
- The user wants to debug a failure -- use `systematic-debugging`.
- The user is asking about status -- read OPERATOR_REGISTRY.yaml directly.
- The user has only a vague idea and is not ready to commit to a spec -- guide them to `npu-operator-intake`.

## Required Inputs

1. **INTAKE.yaml** at `.npu-harness/operators/<op_name>/INTAKE.yaml` (or the user provides equivalent structured information directly).
2. **OPERATOR_REGISTRY.yaml** for reading current operator entries and avoiding conflicts.
3. **Harness defaults** from REQUIREMENTS.md and CONTEXT.md for default tolerances, arch, and schema version.
4. **Operator category knowledge** -- the spec author must understand what constraints are specific to the category (e.g., matmul has cube unit constraints, attention has online softmax requirements).

## Required Outputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` following the extended harness spec schema.
2. **Updated OPERATOR_REGISTRY.yaml** with `spec_authoring_complete: true` (or equivalent status).
3. **Updated STATUS.md** at `.npu-harness/operators/<op_name>/STATUS.md` reflecting the new spec phase.

## Extended Spec Schema

The SPEC must conform to this full schema:

```yaml
schema_version: 2
name: "<op_name>"
category: "<category>"
target_arch: "dav-2201"

summary: "<one-line description>"

math:
  definition: "<formula>"
  equations: []
  notes: []

inputs:
  - name: "<name>"
    dtype: ["<dtype>", ...]
    shape: ["<sym>", ...]
    layout: "<layout>"
    role: "<role>"

outputs:
  - name: "<name>"
    dtype: ["<dtype>", ...]
    shape: ["<sym>", ...]
    layout: "<layout>"

parameters:
  - name: "<name>"
    type: "<float|int|bool>"
    default: <value>
    constraints: "<description>"

shape_symbols:
  <sym>:
    allowed: [<values>]
    min: <int>
    max: <int>

shape_rules:
  - "<rule>"

dtype_rules:
  accumulation: "<float32|same_as_input>"
  output_cast: "<specified_by_output_dtype|same_as_input>"

layout_rules:
  contiguous_required: <true|false>
  supported_layouts: ["<layouts>"]

broadcasting:
  supported: <true|false>
  rules: []

masking:
  supported: <true|false>
  types: ["<types>"]

tiling:
  strategy: "<auto|manual>"
  constraints: []
  workspace: "<optional|required>"

implementation:
  cpu_golden: "<required|optional>"
  ascend_kernel: "<required|optional>"
  direct_launch: "<required|not_applicable>"
  framework_substitution_allowed: false

accuracy:
  float32: { abs: 1.0e-5, rel: 1.0e-5 }
  float16: { abs: 1.0e-2, rel: 1.0e-2 }

performance:
  required: <true|false>
  targets:
    - shape: "<concrete>"
      dtype: "<dtype>"
      device: "dav-2201"
      metric: "<latency_us|bandwidth_gb_s|tflops>"
      target: <value>
      comparison: "<<=|>=|==>"
  tuning_budget:
    max_iterations: 5

tests:
  deterministic_seed: 20260522
  shapes:
    - [<shape1>]
    - [<shape2>]
  edge_cases: []
  invalid_cases: []

benchmark:
  required: <true|false>
  warmup: 10
  iterations: 100

status:
  generated: false
  built: false
  host_tested: false
  npu_tested: false
  accuracy_passed: false
  performance_passed: false
```

## Step-by-Step Procedure

### Step 1: Load and Validate the INTAKE.yaml

Read `.npu-harness/operators/<op_name>/INTAKE.yaml`. Verify that:

- All blocking fields (inputs, outputs, shapes, dtypes, math) are populated.
- There are no unresolved open questions in `open_questions`.
- If open questions remain, stop and instruct the user to resolve them via `/npu-discuss-op <op_name>`.

### Step 2: Check the Operator Registry

Read `OPERATOR_REGISTRY.yaml`. Verify:

- The operator name is unique.
- The category does not conflict with any architectural constraint.
- No stale entries exist that should be cleaned up first.

### Step 3: Expand Shapes into Complete Test Coverage

For each shape symbol in `shape_symbols`, select a set of concrete values that provide good coverage:

- **Tiny:** the minimum allowed value (usually 1).
- **Small:** a value like 8, 16, or 32.
- **Medium:** a moderate value like 128, 256, 512.
- **Large:** a non-trivial size like 1024, 4096, 8192.
- **Non-power-of-2:** a value like 17, 63, 257 (to exercise tail handling).
- **Singleton:** all dimensions = 1 simultaneously.

At least 3 concrete shape combinations must be listed in `tests.shapes`. At minimum, one tiny, one medium, and one with a non-power-of-2 dimension.

### Step 4: Define Edge Cases and Invalid Cases

**Edge cases** (valid inputs, tricky behavior):
- Zeros, ones, large values, small values.
- `nan` and `inf` where the operator should handle them.
- Constant inputs (all same value).
- Alternating sign inputs.
- Maximum/minimum representable values for the dtype.

**Invalid cases** (should be rejected by host-side validation):
- Dimension mismatch between inputs.
- Unsupported broadcast forms.
- Violation of `rotary_dim <= H` or `rotary_dim % 2 == 0`.
- Negative or zero dimensions.
- Unsupported dtype combinations.
- Missing required parameters.

### Step 5: Set Accuracy Tolerances per Dtype

Apply these default tolerances unless the operator category requires different values:

| Category | FP32 abs | FP32 rel | FP16 abs | FP16 rel |
|---|---|---|---|---|
| elementwise | 1e-5 | 1e-5 | 1e-2 | 1e-2 |
| reduction | 1e-4 | 1e-4 | 5e-2 | 5e-2 |
| layout | 1e-5 | 1e-5 | 1e-2 | 1e-2 |
| normalization | 1e-4 | 1e-4 | 5e-2 | 5e-2 |
| matmul | 1e-4 | 1e-4 | 1e-1 | 1e-1 |
| attention | 1e-4 | 1e-4 | 1e-1 | 1e-1 |
| fusion | *use the tightest tolerance of fused sub-operations* |
| custom | 1e-4 | 1e-4 | 5e-2 | 5e-2 |

Document any deviation from these defaults in the `math.notes` field.

### Step 6: Write the Tiling and Implementation Constraints

Fill in the `tiling` and `implementation` sections:

- `tiling.strategy`: `auto` for simple elementwise/reductions; `manual` for matmul, attention, and fusions.
- `tiling.constraints`: list any alignment requirements (e.g., "tile size must be multiple of 32 for dav-2201 vector width").
- `tiling.workspace`: `optional` unless the algorithm requires scratch space (e.g., flash attention needs per-row statistics).
- `implementation.cpu_golden`: always `required`.
- `implementation.ascend_kernel`: `required` unless the operator is host-only by design.
- `implementation.direct_launch`: always `required`.
- `implementation.framework_substitution_allowed`: always `false`.

### Step 7: Set the Benchmark Section

- `benchmark.required`: `true` if `performance.required` is `true` or if sanity benchmarks are desired. `false` only if the user explicitly says no benchmarks.
- `benchmark.warmup`: 10 (harness default).
- `benchmark.iterations`: 100 (harness default).

### Step 8: Write the SPEC.yaml File

Write the complete YAML to `specs/<op_name>.yaml`. Validate YAML syntax.

### Step 9: Reject Invalid Specs

Before finalizing, check that the spec is NOT missing any of these critical fields. If any are missing, reject the spec and report the deficiency:

1. **No shapes defined** -- `tests.shapes` must have at least 3 entries.
2. **No dtypes defined** -- `inputs[].dtype` and `outputs[].dtype` must be populated.
3. **No output definitions** -- `outputs` must have at least one entry with name, dtype, and shape.
4. **No correctness checks** -- `accuracy` must have at least one dtype tolerance entry.
5. **No tolerance** -- each declared dtype must have both `abs` and `rel` tolerance values.
6. **Performance target missing when requested** -- if `performance.required: true`, `performance.targets` must have at least one entry.
7. **No skip/blocker policy** -- implied by `implementation.ascend_kernel: required` and `framework_substitution_allowed: false`. Explicitly verify that a missing-NPU skip path is documented in the notes.

### Step 10: Update OPERATOR_REGISTRY.yaml

Set or update:
```yaml
operators:
  <op_name>:
    spec: "specs/<op_name>.yaml"
    spec_complete: true
    category: "<category>"
```

### Step 11: Update STATUS.md

Create or update `.npu-harness/operators/<op_name>/STATUS.md`:

```markdown
# Status: <op_name>

| Phase | Status | Date |
|-------|--------|------|
| intake | complete | <date> |
| spec | complete | <date> |
| plan | pending | -- |
| implement | pending | -- |
| build | pending | -- |
| test | pending | -- |
| benchmark | pending | -- |
| ship | pending | -- |
```

## Verification Checklist

Before declaring spec authoring complete, verify each item:

- [ ] `schema_version` is set (value: 2).
- [ ] `name`, `category`, `target_arch` are present and correct.
- [ ] `target_arch` is `dav-2201`.
- [ ] `summary` is a one-line description of the operator.
- [ ] `math.definition` is precise enough for implementation.
- [ ] Every input has `name`, `dtype` (list), `shape` (list), `layout`, `role`.
- [ ] Every output has `name`, `dtype` (list), `shape` (list), `layout`.
- [ ] All shape symbols in shapes are defined in `shape_symbols`.
- [ ] `shape_symbols` entries have `allowed` values or `min`/`max` ranges.
- [ ] `shape_rules` are explicit and internally consistent.
- [ ] `dtype_rules.accumulation` is set (default: `float32`).
- [ ] `dtype_rules.output_cast` is set.
- [ ] `layout_rules` specify contiguous requirement and supported layouts.
- [ ] `broadcasting.supported` is set to true or false.
- [ ] If broadcasting is supported, `broadcasting.rules` are populated.
- [ ] `masking.supported` is set to true or false.
- [ ] If masking is supported, `masking.types` are populated.
- [ ] `tiling.strategy` is set to `auto` or `manual`.
- [ ] `implementation.cpu_golden` is `required`.
- [ ] `implementation.ascend_kernel` is `required`.
- [ ] `implementation.direct_launch` is `required`.
- [ ] `implementation.framework_substitution_allowed` is `false`.
- [ ] `accuracy` has entries for every declared dtype with `abs` and `rel` values.
- [ ] `performance.required` is set.
- [ ] If `performance.required: true`, `performance.targets` has at least one entry.
- [ ] `tests.deterministic_seed` is `20260522`.
- [ ] `tests.shapes` has at least 3 entries covering tiny, medium, and non-power-of-2.
- [ ] `tests.edge_cases` are specified.
- [ ] `tests.invalid_cases` are specified.
- [ ] `benchmark.warmup` is 10.
- [ ] `benchmark.iterations` is 100.
- [ ] `status.*` fields are initialized to `false`.
- [ ] SPEC.yaml is valid YAML (parseable).
- [ ] OPERATOR_REGISTRY.yaml is updated.
- [ ] STATUS.md is updated.
