# Architecture

## Two-Layer Architecture

### Layer A: Agentic Harness Layer
Turns Claude Code into a repeatable NPU operator engineering agent.

**Components:**
- `CLAUDE.md` — Harness agent instructions
- `AGENTS.md` — Agent role definitions (10 roles)
- `.claude/skills/` — 10 domain-specific skills
- `.claude/commands/` — 11 slash commands
- `.npu-harness/` — Persistent state (survives context resets)
- `OPERATOR_REGISTRY.yaml` — Source of truth for operator status
- `scripts/harness.py` — CLI for all harness operations

### Layer B: Concrete NPU Engineering Layer
Real C++/Ascend C project with build, test, benchmark infrastructure.

**Components:**
- `CMakeLists.txt` — Root build configuration
- `cmake/` — CMake modules (Ascend, GTest, CompilerOptions, OperatorTarget)
- `include/harness/` — C++ utility headers
- `src/` — C++ utility implementations
- `specs/` — Operator spec YAML files
- `ops/` — Operator implementations
- `tests/` — GTest test files

## Operator Workflow

```
intake -> spec -> discuss -> plan -> generate -> implement -> build -> test -> benchmark -> tune -> verify -> ship
```

## Build System

### Host-Only Mode
- No CANN/NPU dependency
- Builds CPU golden, GTest tests
- Runtime checks skip NPU gracefully
- Command: `cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF`

### NPU Mode
- Requires CANN and NPU device
- Builds Ascend C kernels with `<<<>>>` direct launch
- Full integration tests
- Command: `cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201`

## Operator Structure

Each operator lives in `ops/<op_name>/`:
- `golden/` — CPU reference implementation (always required)
- `kernel/` — Ascend C kernel (`<<<>>>` launch)
- `host/` — Tiling design and shape logic
- `runner/` — Unified runner (dispatches CPU or NPU)
- `bench/` — Performance benchmark
- `CMakeLists.txt` — Build target

## Target Architecture

Default NPU arch: **dav-2201** (Ascend 910B/910C)
