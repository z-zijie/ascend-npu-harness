# /npu-new-op

Accept a natural-language operator requirement and start the complete operator workflow.

## Usage

```
/npu-new-op <<<Implement fused SwiGLU + quantization for shape [B,S,H], dtype fp16 input, int8 output, per-channel scales.>>>
```

The argument is a free-form natural-language description of the operator. It can be a few words or several paragraphs. The Harness will extract structure from it.

## Arguments

- `$ARGUMENTS` — The full natural-language operator requirement. Everything after `<<<` until `>>>`.

## Preconditions

Before starting, verify:

- `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `CONTEXT.md` exist
- `.npu-harness/config.yaml` exists
- `scripts/` directory contains `harness.py`, `detect_ascend_env.py`, and `new_op.py`

If any preconditions are missing, run `/npu-map-codebase` first (or `/npu-new-project` for first-time setup).

## Workflow

### Step 1: Parse the Requirement

Read the natural-language description and extract structured information using the `npu-operator-intake` skill:

- Operator name (infer a concise unique name in snake_case)
- Category (elementwise, reduction, layout, matmul, normalization, attention, fusion, custom)
- Mathematical definition (formula or description)
- Input tensors (names, dtypes, shapes, layouts)
- Output tensors (names, dtypes, shapes, layouts)
- Scalar/host parameters (names, types, defaults)
- Shape rules and constraints
- Broadcasting rules
- Masking requirements
- Dtype rules (accumulation dtype, output cast behavior)
- Accuracy expectations
- Performance targets (if mentioned)
- Target architecture (default: dav-2201)

### Step 2: Clarify Ambiguity

If critical information is missing (see list below), ask the user. If a missing detail can be inferred safely, infer it and record the assumption.

Critical missing information that MUST be asked:
- Unresolvable ambiguity about output shape
- Conflicting dtype requirements
- Undefined behavior for edge cases that affect correctness
- Unclear mathematical definition (operator cannot be implemented without it)

Details that may be INFERRED and recorded:
- Default dtype = float16 if not specified
- Default layout = contiguous if not specified
- Default tolerance = Harness defaults from config
- Default target arch = dav-2201
- Test shapes can be derived from shape symbols

### Step 3: Author the Spec

Use the `npu-spec-authoring` skill to produce the SPEC file.

Create `.npu-harness/operators/<op_name>/SPEC.yaml` with the full extended schema:

```yaml
schema_version: 1
name: <op_name>
category: <category>
target_arch: dav-2201

summary: "<one-line description>"
math:
  definition: "<mathematical definition>"
  equations: []
  notes: []

inputs:
  - name: <name>
    dtype: [<dtypes>]
    shape: [<shape symbols>]
    layout: contiguous
    role: <role>

outputs:
  - name: <name>
    dtype: [<dtypes>]
    shape: [<shape symbols>]
    layout: contiguous

parameters:
  - name: <name>
    type: <type>
    default: <value>

shape_symbols: {}
shape_rules: []
dtype_rules:
  accumulation: float32
  output_cast: specified_by_output_dtype

layout_rules:
  contiguous_required: true

broadcasting:
  supported: <true|false>
  rules: []

masking:
  supported: <true|false>
  types: []

tiling:
  strategy: auto
  constraints: []
  workspace: optional

implementation:
  cpu_golden: required
  ascend_kernel: required
  direct_launch: required
  framework_substitution_allowed: false

accuracy:
  float32:
    abs: 1.0e-5
    rel: 1.0e-5
  float16:
    abs: 1.0e-2
    rel: 1.0e-2

performance:
  required: <true|false>
  targets: []
  tuning_budget:
    max_iterations: 5

tests:
  deterministic_seed: <generated from date>
  shapes: []
  edge_cases: []
  invalid_cases: []

benchmark:
  required: <true|false>
  warmup: 10
  iterations: 100
```

Also create (or symlink) `specs/<op_name>.yaml` for the public-facing spec. This may be a copy of or a reference to the canonical `.npu-harness/operators/<op_name>/SPEC.yaml`.

### Step 4: Enforce Spec Quality

Reject the spec if ANY of these are missing:

- `name` field
- `inputs` with dtype and shape for each input
- `outputs` with dtype and shape for each output
- `accuracy` section with tolerances
- `math.definition` (the core mathematical operation)

If `performance.required: true`, also reject if:
- `performance.targets` is empty
- No metric is specified
- No comparison operator is specified

### Step 5: Generate Implementation Scaffold

Run the spec generator:

```bash
python3 scripts/generate_op_from_spec.py specs/<op_name>.yaml
```

or, if using the harness CLI:

```bash
python3 scripts/harness.py new-op --spec specs/<op_name>.yaml
```

This must produce:

```
ops/<op_name>/
├── CMakeLists.txt
├── kernel/
│   └── <op_name>_kernel.cpp
├── host/
│   ├── <op_name>_tiling.hpp
│   └── <op_name>_tiling.cpp
├── runner/
│   ├── <op_name>_runner.hpp
│   └── <op_name>_runner.cpp
├── golden/
│   ├── <op_name>_golden.hpp
│   └── <op_name>_golden.cpp
├── bench/
│   └── bench_<op_name>.cpp
└── README.md

tests/test_<op_name>.cpp
docs/operators/<op_name>.md
```

### Step 6: Create Operator State Files

Create under `.npu-harness/operators/<op_name>/`:

**STATUS.md:**
```markdown
# Status: <op_name>

Last updated: <today>

| Phase | Status | Notes |
|-------|--------|-------|
| spec | done | |
| plan | pending | |
| golden | pending | |
| shape_validation | pending | |
| tiling | pending | |
| npu_kernel | pending | |
| runner | pending | |
| host_tests | pending | |
| npu_tests | pending | |
| accuracy | pending | |
| benchmark | pending | |
| ship | pending | |
```

**PLAN.md** — Empty template, to be filled by `/npu-plan-op`.

**DECISIONS.md:**
```markdown
# Decisions: <op_name>

## Inferred Assumptions
- <list each assumption made during intake>
```

### Step 7: Update Registry

Add the operator to `OPERATOR_REGISTRY.yaml`:

```yaml
operators:
  <op_name>:
    spec: specs/<op_name>.yaml
    status_file: .npu-harness/operators/<op_name>/STATUS.md
    category: <category>
    generated: true
    built_host: false
    tested_host: false
    built_npu: false
    tested_npu: skipped
    npu_skip_reason: "Not yet implemented"
    accuracy: pending
    performance: pending
```

### Step 8: Report

Print a summary:

```
Operator <op_name> intake complete.

Created:
  .npu-harness/operators/<op_name>/SPEC.yaml
  .npu-harness/operators/<op_name>/PLAN.md
  .npu-harness/operators/<op_name>/STATUS.md
  .npu-harness/operators/<op_name>/DECISIONS.md
  specs/<op_name>.yaml
  ops/<op_name>/* (generated scaffold)
  tests/test_<op_name>.cpp
  docs/operators/<op_name>.md

Next command: /npu-discuss-op <op_name>
```

## Rules

1. No operator is accepted without a spec.
2. No spec is accepted without correctness criteria (accuracy section with tolerances).
3. No spec is accepted without dtype/shape requirements for every input and output.
4. If performance matters (`performance.required: true`), no spec is accepted without a measurable performance contract (at least one target with metric, target value, and comparison operator).
5. All inferred assumptions must be recorded in DECISIONS.md.

## Verification Checklist

Before declaring `/npu-new-op` complete:

- [ ] Natural-language requirement parsed into structured fields
- [ ] Operator name assigned (unique, snake_case)
- [ ] Category assigned
- [ ] SPEC.yaml created with complete schema
- [ ] Spec passes all quality gates (name, inputs, outputs, accuracy, math)
- [ ] Performance contract present if required
- [ ] `specs/<op_name>.yaml` created
- [ ] Implementation scaffold generated under `ops/<op_name>/`
- [ ] Test file `tests/test_<op_name>.cpp` created
- [ ] Documentation `docs/operators/<op_name>.md` created
- [ ] STATUS.md created with all phases set to pending
- [ ] DECISIONS.md created with inferred assumptions
- [ ] PLAN.md created (empty, ready for planning)
- [ ] OPERATOR_REGISTRY.yaml updated with new operator entry
