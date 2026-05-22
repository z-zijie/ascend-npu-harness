# NPU Operator Development Harness

A reusable, spec-driven, stateful, agentic engineering system for Ascend NPU operator development.

**Target:** Ascend 910B/910C (dav-2201)
**Style:** C++17 + GTest + CMake + Ninja + CTest + Ascend C + direct `<<<>>>` launch

## Quick Start

```bash
# Initialize
python3 scripts/harness.py init

# Detect environment
python3 scripts/detect_ascend_env.py

# Build (host-only, always works)
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
ctest --test-dir build/host --output-on-failure

# Build (NPU, requires CANN)
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu
ctest --test-dir build/npu --output-on-failure
```

## Operator Workflow

```
intake -> spec -> discuss -> plan -> generate -> implement -> build -> test -> benchmark -> tune -> verify -> ship
```

## Status

```bash
python3 scripts/harness.py status
```

## Adding a New Operator

```bash
# Create spec from template
python3 scripts/new_op.py my_operator elementwise

# Edit the spec
vim specs/my_operator.yaml

# Generate skeleton
python3 scripts/generate_op_from_spec.py specs/my_operator.yaml

# Implement golden, kernel, runner
# ... edit ops/my_operator/golden/my_operator_golden.cpp ...

# Build and test
python3 scripts/harness.py build --mode host
python3 scripts/harness.py test --mode host --op my_operator
```

## Architecture

Two-layer design:
- **Layer A:** Agentic Harness (skills, commands, state files, CLI)
- **Layer B:** Concrete NPU Engineering (CMake, C++ utilities, operators, tests)

See `docs/architecture.md` for details.

## Operator Ladder

| # | Operator | Category |
|---|----------|----------|
| 1 | elementwise_add | elementwise |
| 2 | axpy_scalar_affine | elementwise |
| 3 | broadcast_binary | elementwise |
| 4 | activation_gelu_silu | elementwise |
| 5 | reduce_sum_lastdim | reduction |
| 6 | transpose_2d_4d | layout |
| 7 | layer_norm | normalization |
| 8 | matmul_bias | matmul |
| 9 | rms_norm_rope_fused | fusion |
| 10 | flash_attention_fwd_fused | attention |
