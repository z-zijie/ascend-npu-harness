# PROJECT.md — NPU Operator Development Harness

## Mission

A reusable, spec-driven, stateful, agentic engineering system for Ascend NPU operator development.

The Harness drives the complete operator workflow:
```
intake -> spec -> discuss -> plan -> generate -> implement -> build -> test -> benchmark -> tune -> verify -> ship
```

## Target Hardware

- NPU Architecture: **dav-2201** (Ascend 910B / 910C class)
- Default device: Ascend 910B/C
- Compiler: Ascend C (ccec)

## Project Style

- **Language:** C++17, Ascend C
- **Build:** CMake + Ninja
- **Test:** GTest + CTest
- **Launch:** Direct `<<<>>>` kernel launch
- **Specs:** YAML
- **Golden:** C++ CPU reference implementations

## Two-Layer Architecture

### Layer A: Agentic Harness Layer
Turns Claude Code into a repeatable NPU operator engineering agent.
- Project-level Claude instructions (CLAUDE.md)
- Reusable skills (.claude/skills/)
- Slash-command workflows (.claude/commands/)
- Persistent state files (.npu-harness/)
- Operator registry (OPERATOR_REGISTRY.yaml)
- Harness CLI (scripts/harness.py)

### Layer B: Concrete NPU Engineering Layer
Real C++/Ascend C project with build, test, benchmark infrastructure.
- Host-only build (no CANN required)
- Optional CANN/NPU build
- CMake/Ninja integration
- GTest/CTest
- CPU golden references
- Accuracy utilities
- Ascend runtime helpers
- Direct `<<<>>>` launch path
- Operator registry
- Spec-driven operator generator
- Benchmarks and performance reporting

## 10-Operator Bootstrap Ladder

| # | Operator | Category | Complexity |
|---|----------|----------|------------|
| 1 | elementwise_add | elementwise | Basic |
| 2 | axpy_scalar_affine | elementwise | Scalar params |
| 3 | broadcast_binary | elementwise | Broadcasting |
| 4 | activation_gelu_silu | elementwise | Math functions |
| 5 | reduce_sum_lastdim | reduction | Reduction |
| 6 | transpose_2d_4d | layout | Layout transform |
| 7 | layer_norm | normalization | Multi-stage |
| 8 | matmul_bias | matmul | Matrix multiply |
| 9 | rms_norm_rope_fused | fusion | LLM fusion |
| 10 | flash_attention_fwd_fused | attention | Full attention |

## Quality Bar

- Production-oriented implementation
- Explicit configuration, not hardcoded paths
- Precise diagnostics, not silent failures
- Deterministic tests with printed seeds
- Small reusable utilities
- Clear CMake targets and CTest labels
- RAII for runtime resources
- Correctness before speed
- Stable host-only mode always
