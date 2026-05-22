# CLAUDE_NPU_HARNESS_TASK.md

## 0. Revised mission: build an agentic NPU operator-development Harness

You are Claude Code acting as a senior AI systems engineer, C++/CMake engineer, Ascend C/NPU kernel engineer, performance engineer, and agentic workflow designer.

This task is **not** merely to implement a few example Ascend C operators.

This task is to build a reusable **NPU Operator Development Harness**: a domain-specific, spec-driven, stateful, agentic engineering system that lets the user submit arbitrary future NPU operator requirements and have the Harness drive the complete workflow from requirement intake to functional correctness, NPU implementation, performance validation, tuning, diagnostics, and shipping.

The Harness should be structurally similar in intent to agentic workflow systems such as:

```text
https://github.com/obra/superpowers
https://github.com/gsd-build/get-shit-done
```

Do not copy those projects. Extract the engineering pattern:

1. composable skills,
2. explicit command entry points,
3. durable context/state files,
4. spec-first development,
5. phase-based planning,
6. isolated execution units,
7. test-driven implementation,
8. systematic debugging,
9. verification before completion,
10. status tracking,
11. repeatable workflows across sessions.

The result should feel like **Superpowers/GSD for Ascend NPU operator development**.

The target hardware class is Ascend 910B / Ascend 910C with target NPU architecture:

```text
dav-2201
```

The required project style is:

```text
C++ + GTest + CMake + Ninja + CTest + Ascend C + direct <<<>>> launch
```

The required agentic workflow style is:

```text
intake -> spec -> discuss -> plan -> generate -> implement -> build -> test -> benchmark -> tune -> verify -> ship
```

The final proof operator remains:

```text
flash_attention_fwd_fused
```

The final success criterion is stronger than the previous version:

> After this Harness is built, the user must be able to come back with a new arbitrary operator requirement and use the Harness to generate, implement, verify, benchmark, tune, and ship a functionally correct and performance-targeted Ascend NPU operator, or receive a precise evidence-based blocker/performance-gap report.

---

## 0.1 Two-layer architecture

Build two layers.

### Layer A: Agentic Harness layer

This layer turns Claude Code into a repeatable NPU operator engineering agent.

It must include:

1. project-level Claude instructions,
2. reusable skills,
3. slash-command workflows,
4. persistent state files,
5. operator requirement intake,
6. operator spec generation,
7. implementation planning,
8. code generation,
9. correctness gates,
10. performance gates,
11. systematic debugging playbooks,
12. review checklists,
13. shipping reports.

This layer must survive context resets. The main Claude context window must not be the only source of truth.

### Layer B: Concrete NPU engineering layer

This layer provides the real C++/Ascend C project.

It must include:

1. host-only build,
2. optional CANN/NPU build,
3. CMake/Ninja integration,
4. GTest/CTest,
5. CPU golden references,
6. accuracy utilities,
7. Ascend runtime helpers,
8. direct `<<<>>>` launch path,
9. operator registry,
10. spec-driven operator generator,
11. benchmarks,
12. performance reporting,
13. diagnostics.

---

## 0.2 Non-negotiable future-operator workflow

After the Harness is complete, the user should be able to invoke a workflow such as:

```text
/npu-new-op <<<Implement fused SwiGLU + quantization for shape [B,S,H], dtype fp16 input, int8 output, per-channel scales, target latency under X us on dav-2201.>>>
```

or:

```bash
claude -p --dangerously-skip-permissions "/npu-new-op <<<...new operator requirement...>>>"
```

The Harness must then drive the full sequence:

1. clarify only truly missing details,
2. write or update an operator spec,
3. record assumptions and decisions,
4. produce a concrete implementation plan,
5. generate files,
6. implement CPU golden,
7. implement shape validation,
8. implement tiling,
9. implement Ascend C kernel,
10. implement direct launch runner,
11. build with CMake/Ninja,
12. run GTest/CTest,
13. compare accuracy,
14. benchmark performance,
15. tune if performance fails,
16. update the operator registry,
17. produce an evidence-based final report.

A new operator is accepted only if:

```text
functional correctness passes
and
performance target passes when a target is specified
```

If a requirement cannot be satisfied, the Harness must produce one of:

```text
precise dependency blocker
precise hardware/runtime blocker
precise API blocker
precise accuracy failure report
precise performance gap report
```

It must not silently pass incomplete work.

---

## 0.3 Required agentic repository additions

In addition to the concrete C++/NPU files described later, create or adapt these files and directories.

```text
.
├── CLAUDE.md
├── AGENTS.md
├── PROJECT.md
├── REQUIREMENTS.md
├── ROADMAP.md
├── STATE.md
├── CONTEXT.md
├── OPERATOR_REGISTRY.yaml
├── .claude/
│   ├── commands/
│   │   ├── npu-map-codebase.md
│   │   ├── npu-new-project.md
│   │   ├── npu-new-op.md
│   │   ├── npu-discuss-op.md
│   │   ├── npu-plan-op.md
│   │   ├── npu-execute-op.md
│   │   ├── npu-verify-op.md
│   │   ├── npu-tune-op.md
│   │   ├── npu-ship-op.md
│   │   ├── npu-status.md
│   │   └── npu-doctor.md
│   └── skills/
│       ├── npu-operator-intake/
│       │   └── SKILL.md
│       ├── npu-spec-authoring/
│       │   └── SKILL.md
│       ├── ascend-c-kernel-implementation/
│       │   └── SKILL.md
│       ├── npu-tiling-design/
│       │   └── SKILL.md
│       ├── cpu-golden-authoring/
│       │   └── SKILL.md
│       ├── gtest-verification/
│       │   └── SKILL.md
│       ├── performance-benchmarking/
│       │   └── SKILL.md
│       ├── systematic-debugging/
│       │   └── SKILL.md
│       ├── verification-before-completion/
│       │   └── SKILL.md
│       └── npu-shipping/
│           └── SKILL.md
├── .npu-harness/
│   ├── config.yaml
│   ├── state/
│   │   ├── PROJECT.md
│   │   ├── REQUIREMENTS.md
│   │   ├── ROADMAP.md
│   │   ├── STATE.md
│   │   └── CONTEXT.md
│   ├── operators/
│   │   └── <op_name>/
│   │       ├── SPEC.yaml
│   │       ├── PLAN.md
│   │       ├── DECISIONS.md
│   │       ├── STATUS.md
│   │       ├── RESULTS.md
│   │       ├── PERF.md
│   │       └── BLOCKERS.md
│   ├── runs/
│   │   └── <timestamp>/
│   │       ├── commands.log
│   │       ├── build.log
│   │       ├── ctest.log
│   │       ├── accuracy.json
│   │       ├── benchmark.json
│   │       └── report.md
│   └── templates/
│       ├── operator_spec.yaml
│       ├── operator_plan.md
│       ├── operator_status.md
│       └── final_report.md
```

The existing concrete C++/NPU layout described later still applies. These agentic files are additional mandatory requirements, not replacements.

---

## 0.4 `CLAUDE.md` requirements

Create `CLAUDE.md`.

It must instruct Claude Code to behave as an NPU operator Harness agent.

Required behavior:

```text
You are working inside an NPU operator development Harness.

Before implementing any operator:
1. Read PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md, CONTEXT.md.
2. Read OPERATOR_REGISTRY.yaml.
3. If working on an operator, read .npu-harness/operators/<op>/SPEC.yaml and PLAN.md.
4. Use the relevant skills under .claude/skills.
5. Do not start with code until the operator contract is clear.
6. Always produce CPU golden first or alongside the kernel.
7. Always add GTest coverage.
8. Always run host tests.
9. If NPU is available, run NPU tests.
10. If performance target exists, run benchmarks and compare to target.
11. Do not declare success without evidence.
12. Do not treat missing NPU as host-only failure.
13. Do not hide blockers.
14. Default NPU arch is dav-2201.
```

`CLAUDE.md` must also define the default operator workflow:

```text
intake -> spec -> discuss -> plan -> execute -> verify -> tune -> ship
```

---

## 0.5 `AGENTS.md` requirements

Create `AGENTS.md`.

It must define the roles used by the Harness:

1. operator requirement analyst,
2. spec author,
3. tiling designer,
4. Ascend C kernel engineer,
5. C++ golden engineer,
6. test engineer,
7. performance engineer,
8. systematic debugger,
9. reviewer,
10. shipping reporter.

These roles may be implemented as instructions rather than actual separate subagents, but the workflow must explicitly use role boundaries.

Each role must define:

1. responsibility,
2. inputs,
3. outputs,
4. failure modes,
5. verification checklist.

---

## 0.6 Slash-command interface

Implement project slash commands under `.claude/commands`.

If the local Claude Code version prefers `.claude/skills/<name>/SKILL.md`, still create `.claude/skills` as the primary autonomous interface and `.claude/commands` as the explicit command interface.

### `/npu-map-codebase`

Purpose:

```text
Analyze repository, detect build system, detect CANN environment, detect existing operators, update STATE.md and OPERATOR_REGISTRY.yaml.
```

Must run or instruct Claude to run:

```bash
pwd
find . -maxdepth 4 -type f | sort
python3 scripts/detect_ascend_env.py || true
python3 scripts/harness.py status || true
```

Must update:

```text
STATE.md
CONTEXT.md
OPERATOR_REGISTRY.yaml
.npu-harness/runs/<timestamp>/report.md
```

### `/npu-new-project`

Purpose:

```text
Initialize the Harness state and baseline project files.
```

Must create:

```text
PROJECT.md
REQUIREMENTS.md
ROADMAP.md
STATE.md
CONTEXT.md
.npu-harness/config.yaml
.npu-harness/state/*
```

Must verify host-only baseline:

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

### `/npu-new-op <<<...>>>`

Purpose:

```text
Accept a natural-language operator requirement and start the complete operator workflow.
```

Must produce:

```text
.npu-harness/operators/<op_name>/SPEC.yaml
.npu-harness/operators/<op_name>/PLAN.md
.npu-harness/operators/<op_name>/STATUS.md
specs/<op_name>.yaml
ops/<op_name>/*
tests/test_<op_name>.cpp
docs/operators/<op_name>.md
```

Must enforce:

```text
No operator is accepted without a spec.
No spec is accepted without correctness criteria.
No spec is accepted without dtype/shape requirements.
If performance matters, no spec is accepted without a measurable performance contract.
```

### `/npu-discuss-op <op_name>`

Purpose:

```text
Resolve ambiguity before planning.
```

If a missing detail can be inferred safely, infer it and record the assumption in:

```text
.npu-harness/operators/<op_name>/DECISIONS.md
```

Ask the user only for high-impact missing details that materially affect correctness, ABI, shape rules, dtype behavior, memory format, or performance target.

### `/npu-plan-op <op_name>`

Purpose:

```text
Create an implementation plan that can be executed in small tasks.
```

The plan must include:

1. file paths,
2. implementation steps,
3. test steps,
4. expected failures before implementation,
5. verification commands,
6. performance plan,
7. blocker detection.

### `/npu-execute-op <op_name>`

Purpose:

```text
Implement the operator according to SPEC.yaml and PLAN.md.
```

Must follow:

```text
RED -> GREEN -> REFACTOR
```

At minimum:

1. write or verify failing tests,
2. implement CPU golden,
3. implement shape validation,
4. implement tiling validation,
5. implement NPU kernel path,
6. implement runner,
7. run focused tests,
8. update STATUS.md.

### `/npu-verify-op <op_name>`

Purpose:

```text
Prove correctness.
```

Must run:

```bash
cmake --build build/host
ctest --test-dir build/host -R <op_name> --output-on-failure
```

If NPU build exists:

```bash
cmake --build build/npu
ctest --test-dir build/npu -R <op_name> --output-on-failure
```

Must emit:

```text
.npu-harness/operators/<op_name>/RESULTS.md
.npu-harness/runs/<timestamp>/accuracy.json
```

### `/npu-tune-op <op_name>`

Purpose:

```text
Benchmark and tune performance until the performance contract passes or a precise performance gap is reported.
```

Must run:

```bash
python3 scripts/harness.py bench <op_name>
python3 scripts/harness.py verify-performance <op_name>
```

Must emit:

```text
.npu-harness/operators/<op_name>/PERF.md
.npu-harness/runs/<timestamp>/benchmark.json
```

### `/npu-ship-op <op_name>`

Purpose:

```text
Finalize the operator, update registry, produce shipping report.
```

Must verify:

1. spec exists,
2. docs exist,
3. CPU golden exists,
4. tests pass,
5. accuracy report exists,
6. benchmark report exists if required,
7. blocker report exists if anything is incomplete.

### `/npu-status`

Purpose:

```text
Print current Harness state and operator status table.
```

Must read:

```text
OPERATOR_REGISTRY.yaml
.npu-harness/operators/*/STATUS.md
.npu-harness/operators/*/RESULTS.md
.npu-harness/operators/*/PERF.md
```

### `/npu-doctor`

Purpose:

```text
Diagnose broken environment, missing CANN, missing Ninja, bad CMake, missing GTest, stale generated files, failed tests, invalid specs.
```

Must run:

```bash
python3 scripts/detect_ascend_env.py
python3 scripts/harness.py doctor
```

---

## 0.7 Skill system requirements

Create project skills under `.claude/skills`.

Each skill must be a directory containing `SKILL.md`.

Each `SKILL.md` must include:

1. skill name,
2. trigger description,
3. when to use,
4. when not to use,
5. required inputs,
6. required outputs,
7. step-by-step procedure,
8. verification checklist.

### `npu-operator-intake`

Use when the user describes a new operator.

Must convert natural language into:

```yaml
name:
category:
math_definition:
inputs:
outputs:
parameters:
shape_rules:
dtype_rules:
layout_rules:
broadcast_rules:
mask_rules:
accuracy_contract:
performance_contract:
target_arch: dav-2201
```

Must ask only essential questions and record assumptions.

### `npu-spec-authoring`

Use when writing or updating operator specs.

Must produce a valid `SPEC.yaml` and update `OPERATOR_REGISTRY.yaml`.

Must reject specs missing:

1. shapes,
2. dtypes,
3. output definitions,
4. correctness checks,
5. tolerance,
6. performance target when requested,
7. skip/blocker policy.

### `ascend-c-kernel-implementation`

Use when implementing Ascend C kernels.

Must enforce:

1. direct `<<<>>>` launch,
2. no framework operator substitution,
3. target arch `dav-2201`,
4. clear tiling,
5. safe tail handling,
6. no out-of-bounds access,
7. documented dtype conversion,
8. graceful skip when CANN unavailable.

### `npu-tiling-design`

Use for tiling and scheduling.

Must produce:

1. tile sizes,
2. block dimensions,
3. alignment constraints,
4. workspace requirements,
5. tail policy,
6. per-core work split,
7. memory movement plan,
8. performance risks.

### `cpu-golden-authoring`

Use for CPU reference implementations.

Must require:

1. C++ implementation,
2. deterministic inputs,
3. float32 accumulation where appropriate,
4. numerically stable softmax/reduction,
5. no reliance on PyTorch as mandatory golden.

### `gtest-verification`

Use for tests.

Must require:

1. shape tests,
2. dtype tests,
3. edge cases,
4. invalid input tests,
5. golden comparison,
6. GTest skip for NPU unavailable,
7. CTest labels.

### `performance-benchmarking`

Use when a performance target exists or for sanity benchmarks.

Must require:

1. warmup,
2. iterations,
3. average/min/max time,
4. bandwidth or FLOPs where meaningful,
5. target comparison,
6. regression report.

### `systematic-debugging`

Use when build, test, accuracy, or benchmark fails.

Must follow:

1. reproduce,
2. isolate,
3. explain root cause,
4. fix one cause,
5. rerun,
6. record evidence.

### `verification-before-completion`

Use before saying done.

Must check:

1. commands actually ran,
2. tests passed,
3. skips are justified,
4. performance target passed or gap reported,
5. generated files are committed or listed,
6. status files are updated.

### `npu-shipping`

Use at the end.

Must produce final report with:

1. status,
2. commands,
3. logs,
4. accuracy,
5. performance,
6. blockers,
7. next steps.

---

## 0.8 Harness CLI requirements

Implement `scripts/harness.py`.

It must provide these subcommands:

```bash
python3 scripts/harness.py init
python3 scripts/harness.py map-codebase
python3 scripts/harness.py status
python3 scripts/harness.py doctor
python3 scripts/harness.py new-op --spec specs/<op>.yaml
python3 scripts/harness.py plan <op_name>
python3 scripts/harness.py generate <op_name>
python3 scripts/harness.py build --mode host
python3 scripts/harness.py build --mode npu
python3 scripts/harness.py test --mode host --op <op_name>
python3 scripts/harness.py test --mode npu --op <op_name>
python3 scripts/harness.py bench <op_name>
python3 scripts/harness.py verify <op_name>
python3 scripts/harness.py verify-performance <op_name>
python3 scripts/harness.py ship <op_name>
```

The CLI must:

1. create timestamped run directories,
2. log commands,
3. capture stdout/stderr,
4. write JSON summaries,
5. update status files,
6. never hide failures,
7. support host-only mode,
8. detect missing CANN/NPU cleanly.

---

## 0.9 Operator registry requirements

Implement `OPERATOR_REGISTRY.yaml`.

Example:

```yaml
schema_version: 1
target_arch_default: dav-2201
operators:
  elementwise_add:
    spec: specs/01_elementwise_add.yaml
    status_file: .npu-harness/operators/elementwise_add/STATUS.md
    category: elementwise
    generated: true
    built_host: true
    tested_host: true
    built_npu: false
    tested_npu: skipped
    npu_skip_reason: "CANN not detected"
    accuracy: pass
    performance: sanity
  flash_attention_fwd_fused:
    spec: specs/10_flash_attention_fwd_fused.yaml
    category: attention
    generated: true
    built_host: true
    tested_host: true
    built_npu: false
    tested_npu: skipped
    npu_skip_reason: "No visible Ascend device"
    accuracy: pass
    performance: pending_npu
```

The registry must be the source of truth for `/npu-status`.

---

## 0.10 Performance contract and tuning loop

The Harness must treat performance as a first-class contract.

Every operator spec must support:

```yaml
performance:
  required: true
  targets:
    - shape: "B=1,S=1024,H=4096"
      dtype: float16
      device: dav-2201
      metric: latency_us
      target: 1000
      comparison: "<="
```

If no explicit target is provided, generate sanity benchmarks but do not claim performance target pass.

If performance fails:

1. inspect benchmark,
2. identify bottleneck hypothesis,
3. change tiling or implementation,
4. rerun benchmark,
5. compare before/after,
6. record decision in `PERF.md`.

Stop only when:

1. target passes, or
2. tuning budget is exhausted and a performance gap report is written, or
3. a concrete external blocker exists.

Performance gap report must include:

```text
target:
measured:
gap:
likely bottleneck:
attempted optimizations:
remaining options:
blocker:
```

---

## 0.11 Extended operator spec schema

The operator spec must support arbitrary future operators.

Use YAML.

Minimum schema:

```yaml
schema_version: 1
name: example_op
category: elementwise | reduction | layout | matmul | normalization | attention | fusion | custom
target_arch: dav-2201

summary: ""
math:
  definition: ""
  equations: []
  notes: []

inputs:
  - name: x
    dtype: [float16, float32]
    shape: [B, S, H]
    layout: contiguous
    role: activation

outputs:
  - name: y
    dtype: [float16, float32]
    shape: [B, S, H]
    layout: contiguous

parameters:
  - name: eps
    type: float
    default: 1.0e-5

shape_symbols:
  B:
    allowed: [1, 2, 4]
  S:
    allowed: [1, 16, 128, 1024]
  H:
    allowed: [64, 128, 768, 4096]

shape_rules:
  - "rotary_dim <= H"
  - "rotary_dim % 2 == 0"

dtype_rules:
  accumulation: float32
  output_cast: specified_by_output_dtype

layout_rules:
  contiguous_required: true
  supported_layouts: [NCHW, NHWC, ND]

broadcasting:
  supported: false
  rules: []

masking:
  supported: false
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
  required: true
  targets:
    - shape: "B=1,S=1024,H=4096"
      dtype: float16
      device: dav-2201
      metric: latency_us
      target: 1000
      comparison: "<="
  tuning_budget:
    max_iterations: 5

tests:
  deterministic_seed: 20260522
  shapes: []
  edge_cases: []
  invalid_cases: []

benchmark:
  required: true
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

---

## 0.12 Revised final report requirement

The final report must include the original operator table plus an explicit Harness layer table.

Required final report structure:

```text
Final status: PASS / PARTIAL / BLOCKED

Harness layer:
- CLAUDE.md:
- AGENTS.md:
- Skills:
- Commands:
- State files:
- Operator registry:
- New operator workflow:
- Performance contract workflow:

C++/NPU layer:
- Build system:
- Host-only support:
- Ascend/NPU support:
- Default NPU arch:
- Direct launch path:
- GTest:
- CTest:
- Generator:
- Accuracy utilities:
- Benchmark utilities:

Commands run:
1. ...
2. ...
3. ...

Build results:
- host configure:
- host build:
- host ctest:
- npu configure:
- npu build:
- npu ctest:

Operator status table:
| # | Operator | Generated | Built | Host tested | NPU tested | Accuracy | Performance | Notes |
|---|----------|-----------|-------|-------------|------------|----------|-------------|-------|
| 1 | elementwise_add | ... | ... | ... | ... | ... | ... | ... |
| 2 | axpy_scalar_affine | ... | ... | ... | ... | ... | ... | ... |
| 3 | broadcast_binary | ... | ... | ... | ... | ... | ... | ... |
| 4 | activation_gelu_silu | ... | ... | ... | ... | ... | ... | ... |
| 5 | reduce_sum_lastdim | ... | ... | ... | ... | ... | ... | ... |
| 6 | transpose_2d_4d | ... | ... | ... | ... | ... | ... | ... |
| 7 | layer_norm | ... | ... | ... | ... | ... | ... | ... |
| 8 | matmul_bias | ... | ... | ... | ... | ... | ... | ... |
| 9 | rms_norm_rope_fused | ... | ... | ... | ... | ... | ... | ... |
|10 | flash_attention_fwd_fused | ... | ... | ... | ... | ... | ... | ... |

Flash Attention proof:
- Implementation status:
- Tiling strategy:
- Online softmax status:
- Mask support:
- Small-shape correctness:
- Medium-shape sanity:
- NPU direct launch:
- Performance:
- Blockers, if any:

Future arbitrary operator workflow proof:
- Command to start new op:
- Spec generation:
- Plan generation:
- Build/test/tune loop:
- State persistence:
- Registry update:
- Pass/fail criteria:

Blockers:
- None
or
- Exact missing dependency / file / library / hardware / API:

Files changed:
- ...

Next commands for the user:
- ...
```

Do not report PASS if:

1. host-only tests fail,
2. default arch is not `dav-2201`,
3. agentic Harness layer is missing,
4. new operator workflow is missing,
5. Flash Attention is only a stub,
6. performance targets are ignored,
7. direct launch path is absent.

---

## 0.13 Relationship to the concrete 10-operator ladder

The 10 operators below are not just examples. They are the bootstrap acceptance matrix proving that the Harness works.

They are still mandatory.

However, they are no longer the whole goal.

The whole goal is:

```text
A reusable agentic Harness that can develop future NPU operators correctly and performance-consciously.
```

The 10-operator ladder is the initial evidence that the Harness is capable.

---

## 0.14 Override rule

If any later section appears narrower than this revised mission, this revised mission wins.

In particular:

1. Build the agentic Harness layer even if a later repository layout only lists C++ files.
2. Build slash commands and skills even if a later section only mentions scripts.
3. Treat performance as a first-class contract even if a later operator only mentions correctness.
4. Treat arbitrary future operator intake as a core requirement.
5. Keep `dav-2201` as the default target architecture everywhere.

---

# Concrete C++/Ascend NPU Harness requirements

The following sections preserve and extend the original concrete NPU engineering requirements.

## 1. Hard constraints

These constraints are mandatory.

1. The default NPU architecture must be `dav-2201`.
2. Do not use `dav-3510` anywhere as the default target architecture.
3. If `dav-3510` appears in any generated file, script, README, CMake preset, test, or command, it is a bug unless it is explicitly marked as a non-default example for other hardware.
4. The primary NPU execution path must use Ascend C direct kernel launch syntax with `<<<>>>`.
5. Do not implement the NPU operator path by calling PyTorch, `torch_npu`, TensorFlow, ONNX Runtime, ACLNN high-level framework ops, or any other framework operator as the real implementation.
6. PyTorch or Python may be used only as an optional extra cross-check, not as the required golden or NPU implementation.
7. Every operator must have a C++ CPU golden reference.
8. Every operator must have GTest tests.
9. Every operator must have CMake/Ninja build integration.
10. Every operator must have CTest registration.
11. If CANN or an NPU device is unavailable, host-only tests must still pass.
12. Missing CANN or missing hardware must result in clear skip messages, not build-system failure for host-only tests.
13. Do not stop after creating README files or scaffolding only.
14. Do not claim the task is complete until configure, build, and tests have been run or precisely blocked.
15. Do not treat operator complexity as a valid reason to omit implementation. Only concrete missing external dependencies, missing headers, missing libraries, missing compiler support, or missing hardware may be reported as blockers.
16. The final answer must include exact commands run, exact results, per-operator status, and blocker details.

---

## 2. Definition of success

The goal is achieved only when all of the following are true.

1. The repository has a clear Harness architecture for custom Ascend C NPU operators.
2. The default target architecture is `dav-2201`.
3. The project supports host-only configure, build, and test.
4. The project supports CANN/NPU configure, build, and test when CANN and NPU hardware are present.
5. NPU kernels are launched through `<<<>>>` direct launch.
6. GTest and CTest are integrated.
7. CMake and Ninja are the primary build path.
8. CPU golden implementations exist and are used for correctness verification.
9. Accuracy comparison reports include max absolute error, max relative error, mismatch count, and the first several mismatch locations.
10. The operator generator can create operator skeletons from spec files.
11. The operator generator is idempotent and does not overwrite manual code unless explicitly forced.
12. At least the first 7 operators in the ladder have complete generated or implemented code paths.
13. `matmul_bias`, `rms_norm_rope_fused`, and `flash_attention_fwd_fused` have complete engineering targets: spec, CMake target, kernel path, tiling path, runner, CPU golden, GTest, CTest, documentation, and benchmark target.
14. `flash_attention_fwd_fused` has either:
    - a working fused forward implementation that passes CPU golden comparison on supported shapes, or
    - a precise blocker report identifying the missing external requirement preventing actual NPU execution.
15. The final report lists the status of every operator.

---

## 3. Expected repository layout

Create or adapt the repository to use this structure.

```text
.
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── cmake/
│   ├── Ascend.cmake
│   ├── CompilerOptions.cmake
│   ├── GTest.cmake
│   └── OperatorTarget.cmake
├── scripts/
│   ├── detect_ascend_env.py
│   ├── generate_op_from_spec.py
│   ├── new_op.py
│   ├── build_host.sh
│   ├── build_npu.sh
│   ├── test_host.sh
│   ├── test_npu.sh
│   └── benchmark_op.sh
├── specs/
│   ├── 01_elementwise_add.yaml
│   ├── 02_axpy_scalar_affine.yaml
│   ├── 03_broadcast_binary.yaml
│   ├── 04_activation_gelu_silu.yaml
│   ├── 05_reduce_sum_lastdim.yaml
│   ├── 06_transpose_2d_4d.yaml
│   ├── 07_layer_norm.yaml
│   ├── 08_matmul_bias.yaml
│   ├── 09_rms_norm_rope_fused.yaml
│   └── 10_flash_attention_fwd_fused.yaml
├── include/
│   └── harness/
│       ├── accuracy.hpp
│       ├── aligned_buffer.hpp
│       ├── ascend_runtime.hpp
│       ├── dtype.hpp
│       ├── logging.hpp
│       ├── shape.hpp
│       ├── skip.hpp
│       ├── tensor.hpp
│       └── test_utils.hpp
├── src/
│   ├── accuracy.cpp
│   ├── ascend_runtime.cpp
│   ├── dtype.cpp
│   ├── shape.cpp
│   └── test_utils.cpp
├── ops/
│   └── <op_name>/
│       ├── CMakeLists.txt
│       ├── kernel/
│       │   └── <op_name>_kernel.cpp
│       ├── host/
│       │   ├── <op_name>_tiling.hpp
│       │   └── <op_name>_tiling.cpp
│       ├── runner/
│       │   ├── <op_name>_runner.hpp
│       │   └── <op_name>_runner.cpp
│       ├── golden/
│       │   ├── <op_name>_golden.hpp
│       │   └── <op_name>_golden.cpp
│       ├── bench/
│       │   └── bench_<op_name>.cpp
│       └── README.md
├── tests/
│   ├── CMakeLists.txt
│   ├── test_elementwise_add.cpp
│   ├── test_axpy_scalar_affine.cpp
│   ├── test_broadcast_binary.cpp
│   ├── test_activation_gelu_silu.cpp
│   ├── test_reduce_sum_lastdim.cpp
│   ├── test_transpose_2d_4d.cpp
│   ├── test_layer_norm.cpp
│   ├── test_matmul_bias.cpp
│   ├── test_rms_norm_rope_fused.cpp
│   └── test_flash_attention_fwd_fused.cpp
├── docs/
│   ├── architecture.md
│   ├── build.md
│   ├── direct_launch.md
│   ├── testing.md
│   ├── troubleshooting.md
│   └── operators/
│       ├── elementwise_add.md
│       ├── axpy_scalar_affine.md
│       ├── broadcast_binary.md
│       ├── activation_gelu_silu.md
│       ├── reduce_sum_lastdim.md
│       ├── transpose_2d_4d.md
│       ├── layer_norm.md
│       ├── matmul_bias.md
│       ├── rms_norm_rope_fused.md
│       └── flash_attention_fwd_fused.md
└── build/
```

If the existing repository has a different layout, preserve useful existing files but converge toward this structure.

---

## 4. CMake and build requirements

The build system must support at least two modes.

### 4.1 Host-only mode

Host-only mode must not require CANN, Ascend C, or NPU hardware.

It must build:

- core Harness library,
- CPU golden implementations,
- GTest tests,
- shape validation,
- tiling validation where possible,
- spec parser or generated spec metadata,
- host-only benchmarks if applicable.

Host-only mode must skip actual NPU tests with clear messages.

Example expected command:

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

### 4.2 Ascend/NPU mode

Ascend/NPU mode should be enabled only when the toolchain is available.

It must build:

- Ascend C kernels,
- host runner code,
- direct launch path,
- NPU integration tests,
- benchmark targets.

Example expected command:

```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu
ctest --test-dir build/npu --output-on-failure
```

### 4.3 Required CMake variables

Provide these CMake variables.

```text
HARNESS_ENABLE_ASCEND       BOOL    default OFF
HARNESS_NPU_ARCH            STRING  default dav-2201
HARNESS_ENABLE_TESTS        BOOL    default ON
HARNESS_ENABLE_BENCHMARKS   BOOL    default ON
HARNESS_ENABLE_WARNINGS     BOOL    default ON
HARNESS_STRICT              BOOL    default OFF
ASCEND_HOME                 PATH    optional
CANN_HOME                   PATH    optional
```

### 4.4 `dav-2201` enforcement

CMake must enforce:

```cmake
set(HARNESS_NPU_ARCH "dav-2201" CACHE STRING "Target NPU architecture")
```

If `HARNESS_ENABLE_ASCEND=ON`, print the selected architecture clearly.

If `HARNESS_NPU_ARCH` is empty, set it to `dav-2201`.

If a user passes another arch, allow it only as an explicit override and print a warning that the default target for this project is `dav-2201`.

---

## 5. Ascend environment detection

Implement `scripts/detect_ascend_env.py`.

It should detect and report:

- whether CANN appears installed,
- candidate `ASCEND_HOME`,
- candidate `CANN_HOME`,
- `ascend-toolkit` path,
- compiler path if available,
- important include paths,
- important library paths,
- whether runtime libraries appear available,
- whether device discovery appears possible,
- recommended CMake command.

It must not fail host-only workflows.

If CANN is missing, it should print a precise message such as:

```text
CANN toolkit not detected. Host-only mode is available.
Use -DHARNESS_ENABLE_ASCEND=OFF for CPU golden and GTest validation.
```

If CANN is detected but device access is unavailable, NPU integration tests should skip at runtime.

---

## 6. Direct launch requirement

The Harness must use a direct Ascend C kernel launch path.

The NPU path should follow the local CANN/Ascend C convention for kernels similar to:

```cpp
// Kernel declaration style depends on the local Ascend C toolchain.
extern "C" __global__ __aicore__
void elementwise_add_kernel(/* GM_ADDR or typed global memory args */, /* tiling args */);
```

and host-side launch similar to:

```cpp
elementwise_add_kernel<<<block_dim, l2ctrl, stream>>>(/* device pointers */, /* scalar args */, /* tiling args */);
```

The exact function attributes, pointer types, stream type, and kernel source extension must be adapted to the local CANN version, but the conceptual requirement is fixed:

- real custom kernel,
- real device memory path,
- direct `<<<>>>` launch,
- no framework operator as implementation.

If the local environment requires a specific wrapper or macro for direct kernel launch, use that wrapper but keep the launch semantics explicit and document it in `docs/direct_launch.md`.

---

## 7. Runtime and test behavior

Implement a runtime utility layer under `include/harness/` and `src/`.

It should provide:

1. Device availability detection.
2. Stream creation and destruction.
3. Device memory allocation and free.
4. Host-to-device copy.
5. Device-to-host copy.
6. Stream synchronization.
7. Error-code to string conversion where possible.
8. RAII wrappers for device resources.
9. Skip helpers for GTest.

NPU tests must use GTest skip semantics when hardware or runtime is unavailable.

Example behavior:

```cpp
if (!harness::ascend::IsAscendRuntimeAvailable()) {
    GTEST_SKIP() << "Ascend runtime is not available; skipping NPU integration test.";
}
if (!harness::ascend::HasUsableDevice()) {
    GTEST_SKIP() << "No usable Ascend NPU device detected; skipping NPU integration test.";
}
```

Host-only tests must never skip due to missing NPU.

---

## 8. Accuracy reporting

Implement reusable accuracy comparison utilities.

The comparison report must include:

- dtype,
- shape,
- element count,
- absolute tolerance,
- relative tolerance,
- maximum absolute error,
- maximum relative error,
- index of maximum absolute error,
- index of maximum relative error,
- mismatch count,
- first N mismatches,
- optional summary statistics.

Suggested API:

```cpp
struct Tolerance {
    double abs_tol;
    double rel_tol;
};

struct Mismatch {
    size_t index;
    double expected;
    double actual;
    double abs_error;
    double rel_error;
};

struct AccuracyReport {
    bool passed;
    size_t num_elements;
    size_t mismatch_count;
    double max_abs_error;
    double max_rel_error;
    size_t max_abs_index;
    size_t max_rel_index;
    std::vector<Mismatch> first_mismatches;
    std::string ToString() const;
};

AccuracyReport CompareArrays(
    std::span<const float> expected,
    std::span<const float> actual,
    Tolerance tolerance,
    size_t max_mismatches_to_record = 16);
```

Use equivalent C++17-compatible code if `std::span` is unavailable.

---

## 9. Operator spec generator

Implement a spec-driven operator generator.

Required entry points:

```bash
python3 scripts/new_op.py <op_name>
python3 scripts/generate_op_from_spec.py specs/01_elementwise_add.yaml
python3 scripts/generate_op_from_spec.py specs/10_flash_attention_fwd_fused.yaml
```

The generator must create or update:

```text
ops/<op_name>/CMakeLists.txt
ops/<op_name>/kernel/<op_name>_kernel.cpp
ops/<op_name>/host/<op_name>_tiling.hpp
ops/<op_name>/host/<op_name>_tiling.cpp
ops/<op_name>/runner/<op_name>_runner.hpp
ops/<op_name>/runner/<op_name>_runner.cpp
ops/<op_name>/golden/<op_name>_golden.hpp
ops/<op_name>/golden/<op_name>_golden.cpp
ops/<op_name>/bench/bench_<op_name>.cpp
ops/<op_name>/README.md
tests/test_<op_name>.cpp
docs/operators/<op_name>.md
```

The generator must also update, or provide instructions to update:

```text
ops/CMakeLists.txt
tests/CMakeLists.txt
docs/operators index if present
```

### 9.1 Generator idempotency

The generator must be idempotent.

Rules:

1. Re-running the generator must not corrupt files.
2. Generated files should contain a clear generated-file marker.
3. Do not overwrite manual implementation files unless `--force` is passed.
4. If a file exists and is not safe to overwrite, print a precise message and leave it untouched.
5. For complex operators, generate TODO blocks with clear blocker labels, but do not stop there if implementation is feasible.
6. Generate valid CMake even if the operator kernel is temporarily stubbed.
7. Generate valid GTest host tests immediately.

### 9.2 Suggested spec schema

Use YAML or JSON. YAML is preferred.

Example fields:

```yaml
name: elementwise_add
category: elementwise
target_arch: dav-2201
dtypes:
  - float32
  - float16
  - bfloat16_optional
inputs:
  - name: x
    shape: same_as_output
  - name: y
    shape: same_as_output
outputs:
  - name: z
    shape: same_as_input
parameters: []
broadcasting: none
requires_reduction: false
requires_matmul: false
requires_online_softmax: false
cpu_golden: required
npu_direct_launch: required
host_tests: required
npu_tests: required_if_available
benchmark: sanity
tolerances:
  float32:
    abs: 1.0e-5
    rel: 1.0e-5
  float16:
    abs: 1.0e-2
    rel: 1.0e-2
test_shapes:
  - [1]
  - [8]
  - [1024]
```

The schema may be extended as needed.

---

## 10. Operator ladder overview

The Harness must support these 10 operators, ordered from easiest to hardest.

1. `elementwise_add`
2. `axpy_scalar_affine`
3. `broadcast_binary`
4. `activation_gelu_silu`
5. `reduce_sum_lastdim`
6. `transpose_2d_4d`
7. `layer_norm`
8. `matmul_bias`
9. `rms_norm_rope_fused`
10. `flash_attention_fwd_fused`

These are not optional examples. They are the graded acceptance matrix.

For each operator, provide:

1. spec file,
2. generated or implemented directory structure,
3. Ascend C kernel,
4. host tiling / shape inference / parameter validation,
5. `<<<>>>` launch path,
6. C++ CPU golden,
7. GTest tests,
8. NPU integration tests when available,
9. graceful skip when unavailable,
10. CMake/Ninja target,
11. CTest label,
12. operator documentation,
13. precision tolerance,
14. performance sanity benchmark where applicable,
15. diagnostic logs for failures.

---

## 11. Operator 1: `elementwise_add`

### 11.1 Requirement

Implement:

```text
z = x + y
```

Inputs:

```text
x: tensor
y: tensor
```

Output:

```text
z: tensor, same shape as x and y
```

Supported shapes:

- 1D,
- 2D,
- 3D,
- 4D,
- contiguous tensors.

Supported dtypes:

- `float32`,
- `float16`,
- `bfloat16` if available.

### 11.2 Tests

Shapes:

```text
[1]
[8]
[1024]
[3, 7]
[4, 16, 64]
[2, 3, 32, 128]
```

Data cases:

- deterministic random,
- zeros,
- ones,
- large values,
- small values,
- `nan`,
- `inf`.

CPU golden:

```cpp
z[i] = x[i] + y[i];
```

Tolerance:

```text
FP32: abs 1e-5, rel 1e-5
FP16: abs 1e-2, rel 1e-2
BF16: reasonable BF16 tolerance if supported
```

### 11.3 Acceptance

Host-only tests pass.

When NPU is available, the `<<<>>>` direct launch output matches the CPU golden output.

---

## 12. Operator 2: `axpy_scalar_affine`

### 12.1 Requirement

Implement:

```text
y = alpha * x + beta
```

Inputs:

```text
x: tensor
alpha: host scalar
beta: host scalar
```

Output:

```text
y: tensor, same shape as x
```

Supported dtypes:

- `float32`,
- `float16`,
- `bfloat16` if available.

This operator validates:

- scalar parameter passing,
- kernel parameter ABI,
- tiling alignment,
- tail handling,
- non-power-of-two lengths.

### 12.2 Tests

Scalar values:

```text
alpha: 0, 1, -1, 0.5, -2.25
beta:  0, 1, -1, 0.5, -2.25
```

Shapes:

```text
[1]
[17]
[1024]
[4097]
[32, 128]
```

Data cases:

- deterministic random,
- edge values,
- positive values,
- negative values,
- mixed signs.

CPU golden:

```cpp
y[i] = alpha * x[i] + beta;
```

### 12.3 Acceptance

The generated runner correctly passes scalar parameters to the kernel.

Tail blocks do not read or write out of bounds.

The NPU result matches CPU golden.

---

## 13. Operator 3: `broadcast_binary`

### 13.1 Requirement

Implement:

```text
z = x * y + bias
```

Input and output shapes:

```text
x:    [B, C, H, W]
y:    [1, C, 1, 1] or [B, C, 1, 1]
bias: [C] or [1, C, 1, 1]
z:    [B, C, H, W]
```

Supported dtypes:

- `float32`,
- `float16`,
- `bfloat16` if available.

The Harness must explicitly model broadcasting rules.

The kernel must not treat broadcast inputs as dense same-shape tensors.

### 13.2 Tests

Shape coverage:

```text
B: 1, 2, 4
C: 1, 3, 16, 64
H: 1, 7, 32, 63
W: 1, 7, 32, 63
```

Data cases:

- deterministic random,
- `C = 1`,
- non-power-of-two spatial dimensions,
- `y` shape `[1, C, 1, 1]`,
- `y` shape `[B, C, 1, 1]`,
- `bias` shape `[C]`,
- `bias` shape `[1, C, 1, 1]`.

CPU golden:

```cpp
z[b, c, h, w] = x[b, c, h, w] * y[b_or_0, c, 0, 0] + bias[c];
```

### 13.3 Acceptance

The spec can express broadcasting.

Generated validation rejects unsupported broadcast forms.

GTest catches layout and broadcasting mistakes.

NPU output matches CPU golden.

---

## 14. Operator 4: `activation_gelu_silu`

### 14.1 Requirement

Implement two activation functions.

GELU with tanh approximation:

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

SiLU:

```text
SiLU(x) = x / (1 + exp(-x))
```

Supported dtypes:

- `float32`,
- `float16`,
- `bfloat16` if available.

### 14.2 Tests

Input range:

```text
[-20, 20]
```

Explicit values:

```text
-20, -10, -5, -1, 0, 1, 5, 10, 20
```

Shapes:

```text
[128]
[1024]
[8, 512]
[2, 16, 128]
```

CPU golden:

- use `std::tanh`,
- use `std::exp`,
- accumulate and compare in float where appropriate.

### 14.3 Acceptance

The Harness supports nonlinear math operations.

Mismatch reports include max absolute error, max relative error, mismatch count, and first N mismatch locations.

NPU output matches CPU golden within dtype-specific tolerances.

---

## 15. Operator 5: `reduce_sum_lastdim`

### 15.1 Requirement

Implement reduction over the last dimension.

Input shapes:

```text
[M, K]
[B, M, K]
```

Output shapes:

```text
[M]
[B, M]
```

Mathematics:

```text
out[..., i] = sum_j x[..., i, j]
```

Supported dtypes:

- `float32`,
- `float16`,
- `bfloat16` if available.

For low precision input, prefer float32 accumulation.

### 15.2 Tests

Dimensions:

```text
K: 1, 7, 32, 128, 1024, 4096
M: 1, 3, 64, 257
B: 1, 2, 8
```

Data cases:

- random,
- all zeros,
- all ones,
- positive and negative mixed values,
- values designed to expose accumulation error.

CPU golden:

```cpp
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc += static_cast<float>(x[row * K + k]);
}
out[row] = acc;
```

### 15.3 Acceptance

The Harness supports reduction operators.

Shape inference handles rank 2 and rank 3.

Tiling handles large K and tail cases.

NPU output matches CPU golden.

---

## 16. Operator 6: `transpose_2d_4d`

### 16.1 Requirement

Implement:

```text
2D transpose: [M, N] -> [N, M]
4D permute:   NCHW -> NHWC
4D permute:   NHWC -> NCHW
```

Supported dtypes:

- `float32`,
- `float16`,
- `int32`.

This operator validates:

- layout mapping,
- global memory access pattern,
- tile copy,
- coalescing where possible,
- exact index computation.

### 16.2 Tests

2D shapes:

```text
[1, 1]
[3, 7]
[32, 32]
[128, 64]
[257, 129]
```

4D shapes:

```text
[1, 3, 224, 224]
[2, 16, 32, 32]
[4, 7, 13, 17]
```

Additional tests:

- NCHW -> NHWC -> NCHW round trip,
- NHWC -> NCHW -> NHWC round trip,
- exact compare for `int32`,
- floating compare for `float32` and `float16`.

### 16.3 Acceptance

The Harness supports layout transformation operators.

The CPU golden is implemented by explicit index mapping.

GTest can localize layout errors.

NPU output matches CPU golden.

---

## 17. Operator 7: `layer_norm`

### 17.1 Requirement

Implement LayerNorm forward over the last dimension.

Given input `x`, `gamma`, `beta`, and `eps`:

```text
mean = reduce_mean(x over last dim)
var  = reduce_mean((x - mean)^2 over last dim)
y    = (x - mean) / sqrt(var + eps) * gamma + beta
```

Input shapes:

```text
x:     [M, K] or [B, S, H]
gamma: [K] or [H]
beta:  [K] or [H]
y:     same shape as x
```

Supported dtypes:

- `float32`,
- `float16`,
- `bfloat16` if available.

Internal mean and variance should use float32 when possible.

### 17.2 Tests

Dimensions:

```text
K/H: 16, 64, 128, 768, 1024, 4096
M:   1, 3, 64, 257
B:   1, 2, 4
S:   1, 16, 128
```

Epsilon:

```text
1e-5
1e-6
```

Data cases:

- random `x`,
- random `gamma`,
- random `beta`,
- constant rows,
- low-variance rows,
- positive and negative mixed values.

### 17.3 Acceptance

The Harness supports multi-input, reduction, and elementwise fusion.

CPU golden is numerically stable.

NPU output matches CPU golden within dtype-specific tolerance.

---

## 18. Operator 8: `matmul_bias`

### 18.1 Requirement

Implement:

```text
C = A @ B + bias
```

Shapes:

```text
A:    [M, K]
B:    [K, N]
bias: [N], optional
C:    [M, N]
```

Supported dtypes:

- input `float16` required,
- input `float32` optional if supported,
- output `float16` or `float32`, selected by spec,
- accumulation should be float32 if possible.

It is acceptable to use Ascend C Matmul high-level APIs if available.

If the local CANN version does not support the required Matmul API, generate a diagnosable fallback or stub and skip NPU tests with a precise blocker message. Host golden must still exist and pass.

### 18.2 Tests

Dimensions:

```text
M: 1, 16, 64, 128, 256
N: 1, 16, 64, 128, 256
K: 1, 16, 64, 128, 256, 1024
```

Required cases:

```text
skinny GEMM: M=1, K=1024, N=64
square GEMM: M=128, K=128, N=128
bias on
bias off
```

CPU golden:

```cpp
for m in M:
  for n in N:
    acc = 0
    for k in K:
      acc += A[m, k] * B[k, n]
    C[m, n] = acc + bias[n] if bias exists else acc
```

### 18.3 Acceptance

The Harness supports matrix multiplication operators.

Tiling configuration is represented.

Dimension validation is correct.

Bias fusion works.

Benchmark target exists.

NPU output matches CPU golden where the environment supports execution.

---

## 19. Operator 9: `rms_norm_rope_fused`

### 19.1 Requirement

Implement an LLM-style fused operator consisting of RMSNorm followed by RoPE on the leading rotary dimensions.

RMSNorm:

```text
rms = sqrt(mean(x^2) + eps)
y   = x / rms * weight
```

RoPE for the first `rotary_dim` hidden dimensions:

```text
out_even = y_even * cos - y_odd * sin
out_odd  = y_even * sin + y_odd * cos
```

Inputs:

```text
x:      [B, S, H]
weight: [H]
cos:    [S, rotary_dim / 2]
sin:    [S, rotary_dim / 2]
eps:    scalar
```

Output:

```text
out: [B, S, H]
```

Rules:

```text
rotary_dim <= H
rotary_dim must be even
dimensions [rotary_dim, H) pass through after RMSNorm
```

Supported dtypes:

- `float16`,
- `bfloat16` if available,
- `float32` golden.

Internal reduction should use float32.

### 19.2 Tests

Dimensions:

```text
B: 1, 2, 4
S: 1, 16, 128, 1024
H: 64, 128, 768, 4096
rotary_dim: 32, 64, 128
```

Epsilon:

```text
1e-5
1e-6
```

Data cases:

- deterministic random `x`,
- deterministic random `weight`,
- generated sinusoidal `cos` and `sin`,
- `S = 1`,
- `rotary_dim = H` when valid,
- `rotary_dim < H`,
- invalid rotary dimensions must be rejected by validation.

### 19.3 Acceptance

The Harness supports LLM fusion patterns.

The operator combines:

- multi-input,
- reduction,
- elementwise scale,
- positional encoding,
- partial-dimension transform,
- validation of shape-dependent parameters,
- complex CPU golden.

NPU output matches CPU golden where the environment supports execution.

---

## 20. Operator 10: `flash_attention_fwd_fused`

### 20.1 Requirement

Implement Flash Attention forward as the final proof target.

This is the highest-difficulty acceptance target.

The implementation must be a custom Ascend C kernel path with host tiling and `<<<>>>` direct launch.

Do not implement this operator by calling a framework attention primitive.

### 20.2 Mathematical definition

Given Q, K, V:

```text
Q: [B, H, S_q, D]
K: [B, H, S_k, D]
V: [B, H, S_k, D_v]
O: [B, H, S_q, D_v]
```

Default:

```text
D_v = D
scale = 1 / sqrt(D)
```

Compute:

```text
scores = Q @ K^T * scale
scores = apply_mask(scores)
P      = softmax(scores)
O      = P @ V
```

Supported masks:

```text
none
causal
additive mask if feasible
```

### 20.3 Core implementation requirements

1. Implement tiled attention.
2. Do not materialize the full `[S_q, S_k]` score matrix into global memory as the main NPU path.
3. Use online softmax or block softmax.
4. Maintain row maximum values.
5. Maintain row sum values.
6. Maintain partial output accumulators.
7. Support causal masking.
8. Support a scale parameter.
9. Support float16 input.
10. Use float32 accumulator where possible.
11. Support parameterized B, H, S, and D.
12. Provide CPU golden reference.
13. CPU golden may materialize the score matrix because it is used only for verification.
14. Provide small-shape strict correctness tests.
15. Provide medium-shape sanity tests.
16. Provide accuracy reports.
17. Provide at least one performance sanity benchmark.
18. Use Ascend C Matmul or Softmax high-level APIs if the local CANN version supports them and if doing so still preserves the custom fused kernel path.
19. If high-level APIs are unavailable, implement a working lower-level version where possible.
20. Do not leave this as pseudocode.
21. Do not leave this as a test-only stub unless a precise external blocker exists.

### 20.4 Online softmax algorithm requirement

For each query row, the implementation should conceptually follow a numerically stable online/block softmax scheme.

For each tile of keys:

```text
scores_tile = Q_row @ K_tile^T * scale
scores_tile = apply_mask(scores_tile)

m_new = max(m_old, max(scores_tile))
exp_old_scale = exp(m_old - m_new)
exp_tile = exp(scores_tile - m_new)

l_new = l_old * exp_old_scale + sum(exp_tile)

acc_new =
    acc_old * exp_old_scale +
    exp_tile @ V_tile

m_old = m_new
l_old = l_new
acc_old = acc_new
```

Final output:

```text
O_row = acc_old / l_old
```

Use equivalent implementation details appropriate for Ascend C.

### 20.5 Tests

Dimensions:

```text
B: 1, 2
H: 1, 4, 8
S_q: 1, 16, 64, 128, 512
S_k: 1, 16, 64, 128, 512
D: 32, 64, 128
D_v: D
```

Masks:

```text
none
causal
```

Dtypes:

```text
float16 required
bfloat16 optional
```

Special cases:

```text
S_q = 1
S_k = 1
S_q != S_k
all-zero Q
all-zero K
large scores
small scores
causal mask correctness
```

Causal rule:

For a causal mask, row `i` may attend only to keys `j <= i` when `S_q == S_k`.

If `S_q != S_k`, define and document the alignment rule. Prefer the standard decoder-compatible alignment:

```text
key_index <= query_index + (S_k - S_q)
```

unless there is a strong reason to use another convention. The selected convention must be implemented identically in CPU golden and NPU path.

### 20.6 CPU golden

CPU golden must be implemented in C++.

It may use straightforward loops:

1. compute scores,
2. apply mask,
3. subtract row max,
4. exponentiate,
5. normalize,
6. multiply by V.

Use float32 math for golden.

### 20.7 Acceptance

The Flash Attention operator is accepted only when:

1. it has a complete spec,
2. it has a CMake target,
3. it has Ascend C kernel code,
4. it has host tiling code,
5. it has a runner,
6. it has CPU golden,
7. it has GTest tests,
8. it has CTest labels,
9. it has documentation,
10. it has a benchmark target,
11. small shapes pass correctness tests,
12. medium shapes pass tolerance tests,
13. NPU execution passes if the environment supports it, or
14. a precise external blocker is reported.

A blocker report must identify exact missing items, for example:

```text
Missing header: <...>
Missing library: ...
Missing compiler: ...
Unsupported CANN API: ...
No visible Ascend device: ...
Runtime initialization failure: ...
```

Generic statements such as "Flash Attention is complex" are not acceptable blockers.

---

## 21. Testing policy

### 21.1 Test categories

Use CTest labels.

Required labels:

```text
host
golden
shape
tiling
npu
integration
benchmark
slow
flash_attention
llm
matmul
reduction
layout
elementwise
```

### 21.2 Required test commands

At minimum, support:

```bash
ctest --test-dir build/host --output-on-failure
ctest --test-dir build/host -L host --output-on-failure
ctest --test-dir build/host -L golden --output-on-failure
```

If Ascend mode is enabled:

```bash
ctest --test-dir build/npu --output-on-failure
ctest --test-dir build/npu -L npu --output-on-failure
ctest --test-dir build/npu -L integration --output-on-failure
```

For Flash Attention:

```bash
ctest --test-dir build/host -L flash_attention --output-on-failure
ctest --test-dir build/npu -L flash_attention --output-on-failure
```

NPU-specific commands must skip cleanly when no NPU is available.

### 21.3 Randomness

All randomized tests must use deterministic seeds.

The seed must be printed when a test fails.

### 21.4 Tolerance

Suggested default tolerances:

```text
float32 elementwise: abs 1e-5, rel 1e-5
float16 elementwise: abs 1e-2, rel 1e-2
float32 reduction: abs 1e-4, rel 1e-4
float16 reduction: abs 5e-2, rel 5e-2
float16 matmul: abs 1e-1, rel 1e-1 unless tighter is feasible
float16 attention: abs 1e-1, rel 1e-1 unless tighter is feasible
```

The exact tolerance may be adjusted per operator, but it must be documented in the operator spec and tests.

---

## 22. Benchmark policy

Benchmarks are sanity checks, not full performance certification.

Each benchmark should report:

- operator name,
- dtype,
- shape,
- number of iterations,
- warmup iterations,
- average time,
- min time,
- max time,
- approximate bandwidth or FLOPs where meaningful,
- whether the run is CPU or NPU,
- NPU arch if applicable.

For example:

```text
operator=elementwise_add dtype=float16 shape=[1048576] device=npu arch=dav-2201 avg_us=...
```

Benchmarks must not be required for host-only correctness success if the machine lacks the needed environment.

---

## 23. Documentation requirements

Write documentation for:

1. project architecture,
2. build instructions,
3. CANN environment detection,
4. direct launch model,
5. how to add a new operator,
6. how to write an operator spec,
7. how to run host tests,
8. how to run NPU tests,
9. how to interpret accuracy failures,
10. troubleshooting.

Each operator document must contain:

- mathematical definition,
- input/output shapes,
- dtype support,
- tiling strategy,
- direct launch notes,
- CPU golden description,
- test coverage,
- expected tolerance,
- benchmark notes,
- known limitations.

---

## 24. Implementation sequence

Use this order unless the existing repository strongly suggests a better sequence.

1. Inspect the repository.
2. Identify existing build files, source files, tests, and scripts.
3. Add or repair top-level CMake/Ninja support.
4. Add host-only GTest support.
5. Add Harness core utilities.
6. Add accuracy utilities.
7. Add Ascend environment detection.
8. Add CANN/NPU optional integration.
9. Add direct-launch abstraction and documentation.
10. Add spec generator.
11. Generate specs for all 10 operators.
12. Implement `elementwise_add`.
13. Implement `axpy_scalar_affine`.
14. Implement `broadcast_binary`.
15. Implement `activation_gelu_silu`.
16. Implement `reduce_sum_lastdim`.
17. Implement `transpose_2d_4d`.
18. Implement `layer_norm`.
19. Implement `matmul_bias`.
20. Implement `rms_norm_rope_fused`.
21. Implement `flash_attention_fwd_fused`.
22. Add tests and CTest labels.
23. Add benchmark targets.
24. Run host configure/build/test.
25. Run Ascend configure/build/test if available.
26. Fix failures.
27. Produce final report.

---

## 25. Required commands to run

Run as many of these as applicable.

### 25.1 Repository inspection

```bash
pwd
find . -maxdepth 3 -type f | sort | sed 's#^\./##'
```

### 25.2 Ascend environment detection

```bash
python3 scripts/detect_ascend_env.py
```

### 25.3 Host build

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

### 25.4 Operator generation

```bash
python3 scripts/generate_op_from_spec.py specs/01_elementwise_add.yaml
python3 scripts/generate_op_from_spec.py specs/02_axpy_scalar_affine.yaml
python3 scripts/generate_op_from_spec.py specs/03_broadcast_binary.yaml
python3 scripts/generate_op_from_spec.py specs/04_activation_gelu_silu.yaml
python3 scripts/generate_op_from_spec.py specs/05_reduce_sum_lastdim.yaml
python3 scripts/generate_op_from_spec.py specs/06_transpose_2d_4d.yaml
python3 scripts/generate_op_from_spec.py specs/07_layer_norm.yaml
python3 scripts/generate_op_from_spec.py specs/08_matmul_bias.yaml
python3 scripts/generate_op_from_spec.py specs/09_rms_norm_rope_fused.yaml
python3 scripts/generate_op_from_spec.py specs/10_flash_attention_fwd_fused.yaml
```

### 25.5 NPU build, only if CANN is available

```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu
ctest --test-dir build/npu --output-on-failure
```

### 25.6 Focused tests

```bash
ctest --test-dir build/host -L host --output-on-failure
ctest --test-dir build/host -L golden --output-on-failure
ctest --test-dir build/host -L flash_attention --output-on-failure
```

If NPU build exists:

```bash
ctest --test-dir build/npu -L npu --output-on-failure
ctest --test-dir build/npu -L integration --output-on-failure
ctest --test-dir build/npu -L flash_attention --output-on-failure
```

---

## 26. Final report format

The final answer must use this structure.

```text
Final status: PASS / PARTIAL / BLOCKED

Repository summary:
- Root:
- Build system:
- Host-only support:
- Ascend/NPU support:
- Default NPU arch:
- Direct launch path:
- GTest:
- CTest:
- Generator:

Commands run:
1. ...
2. ...
3. ...

Build results:
- host configure:
- host build:
- host ctest:
- npu configure:
- npu build:
- npu ctest:

Operator status table:
| # | Operator | Generated | Built | Host tested | NPU tested | Accuracy | Benchmark | Notes |
|---|----------|-----------|-------|-------------|------------|----------|-----------|-------|
| 1 | elementwise_add | ... | ... | ... | ... | ... | ... | ... |
| 2 | axpy_scalar_affine | ... | ... | ... | ... | ... | ... | ... |
| 3 | broadcast_binary | ... | ... | ... | ... | ... | ... | ... |
| 4 | activation_gelu_silu | ... | ... | ... | ... | ... | ... | ... |
| 5 | reduce_sum_lastdim | ... | ... | ... | ... | ... | ... | ... |
| 6 | transpose_2d_4d | ... | ... | ... | ... | ... | ... | ... |
| 7 | layer_norm | ... | ... | ... | ... | ... | ... | ... |
| 8 | matmul_bias | ... | ... | ... | ... | ... | ... | ... |
| 9 | rms_norm_rope_fused | ... | ... | ... | ... | ... | ... | ... |
|10 | flash_attention_fwd_fused | ... | ... | ... | ... | ... | ... | ... |

Flash Attention proof:
- Implementation status:
- Tiling strategy:
- Online softmax status:
- Mask support:
- Small-shape correctness:
- Medium-shape sanity:
- NPU direct launch:
- Blockers, if any:

Blockers:
- None
or
- Exact missing dependency / file / library / hardware / API:

Files changed:
- ...

Next commands for the user:
- ...
```

Do not omit the operator status table.

Do not report PASS if Flash Attention is only a stub.

Do not report PASS if host-only tests fail.

Do not report PASS if the default NPU architecture is not `dav-2201`.

---

## 27. Quality bar

The implementation should be production-oriented.

Avoid:

- fragile shell-only behavior,
- hardcoded absolute paths unless they are clearly optional defaults,
- silent test skipping,
- unclear build errors,
- missing shape validation,
- unbounded memory allocation,
- tests that pass without checking values,
- framework calls pretending to be custom kernels,
- pseudocode-only kernels,
- undocumented tolerances,
- undocumented blockers.

Prefer:

- explicit configuration,
- precise diagnostics,
- deterministic tests,
- small reusable utilities,
- clear CMake targets,
- clear CTest labels,
- RAII for runtime resources,
- generated specs,
- documented limitations,
- correctness before speed,
- stable host-only mode.

---

## 28. Stop condition

Continue working until one of the following is true.

### 28.1 PASS

All mandatory conditions are satisfied, including host tests and the Flash Attention proof target.

### 28.2 PARTIAL

A substantial portion is implemented, host-only tests pass, but some advanced NPU paths are incomplete for clearly stated reasons.

This is acceptable only if the final report is precise and useful.

### 28.3 BLOCKED

Progress is blocked by a concrete external condition.

Valid blockers include:

- no write permission,
- missing compiler,
- missing CMake,
- missing Ninja,
- missing CANN headers,
- missing CANN libraries,
- missing Ascend compiler,
- no visible NPU device,
- runtime initialization failure,
- incompatible CANN API.

Invalid blockers include:

- "the task is complex",
- "Flash Attention is hard",
- "custom operators are difficult",
- "not enough context" when the repository can be inspected,
- "hardware may be missing" without actually checking.

If blocked, produce the final report with exact evidence.

---

## 29. Important reminder

The target architecture for this task is:

```text
dav-2201
```

The final proof operator is:

```text
flash_attention_fwd_fused
```

The required project style is:

```text
C++ + GTest + CMake + Ninja + CTest + Ascend C + direct <<<>>> launch
```

The final goal is:

```text
A stable Harness that can automatically and correctly develop NPU operators for Ascend 910B/C-class dav-2201 targets, from simple elementwise operators up to fused Flash Attention forward.
```
