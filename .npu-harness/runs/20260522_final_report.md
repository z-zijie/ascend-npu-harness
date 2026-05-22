# NPU Operator Development Harness — Final Report

**Date:** 2026-05-22  
**Final status:** PARTIAL (host layer verified 150/150; NPU kernels vectorized, pending AscendC compilation)

---

## Harness Layer Status

| Component | Status | Details |
|-----------|--------|---------|
| CLAUDE.md | PASS | Harness agent instructions with full 14-rule behavior |
| AGENTS.md | PASS | 10 agent roles: analyst, spec author, tiling designer, kernel engineer, golden engineer, test engineer, perf engineer, debugger, reviewer, shipper |
| Skills (10) | PASS | intake, spec-authoring, ascend-c-kernel-implementation, tiling-design, cpu-golden-authoring, gtest-verification, performance-benchmarking, systematic-debugging, verification-before-completion, npu-shipping |
| Commands (11) | PASS | map-codebase, new-project, new-op, discuss-op, plan-op, execute-op, verify-op, tune-op, ship-op, status, doctor |
| State files | PASS | PROJECT, REQUIREMENTS, ROADMAP, STATE, CONTEXT, .npu-harness/config, .npu-harness/state/*, .npu-harness/templates/* |
| Operator registry | PASS | OPERATOR_REGISTRY.yaml with all 10 operators, live status |
| CLI (harness.py) | PASS | 13 subcommands: init, map-codebase, status, doctor, new-op, plan, generate, build, test, bench, verify, verify-performance, ship |
| New operator workflow | PASS | intake -> spec -> plan -> generate -> implement -> build -> test, all stateful |
| Performance contract | PASS | Schema with targets, tuning budget; PERF.md and benchmark.json outputs |

---

## C++/NPU Engineering Layer Status

| Component | Status | Details |
|-----------|--------|---------|
| Build system | PASS | CMake 3.20 + Ninja, host/NPU modes, CMakePresets.json (4 presets) |
| Host-only support | PASS | Full build with 150 tests, all pass |
| Ascend/NPU kernels | VECTORIZED | All 10 kernels rewritten with AscendC vector/Cube APIs |
| NPU kernel APIs used | — | `AscendC::Add`, `AscendC::Mul`, `AscendC::Muls`, `AscendC::Adds`, `AscendC::Sub`, `AscendC::Div`, `AscendC::Exp`, `AscendC::Tanh`, `AscendC::Sqrt`, `AscendC::ReduceSum`, `AscendC::ReduceMean`, `AscendC::MatMul`, `AscendC::DataCopy`, `AscendC::TPipe`, `AscendC::TQue`, `AscendC::GlobalTensor`, `AscendC::LocalTensor` |
| Default NPU arch | PASS | dav-2201 enforced in CMake, specs, kernels |
| Direct launch path | PASS | `extern "C" __global__ __aicore__` + tiling data via `REGISTER_TILING_DEFAULT` |
| GTest/CTest | PASS | 1.14.0, 150 tests, labels: host/golden/npu/elementwise/reduction/layout/matmul/normalization/attention/fusion/llm |
| Generator | PASS | Spec-driven, idempotent, creates full operator skeletons |
| Accuracy utilities | PASS | CompareArrays: max abs/rel error, mismatch count, first N mismatches |
| Zero scalar loops | PASS | No `for(i=0;i<n;i++){z[i]=x[i]+y[i]}` in any kernel |

---

## Commands Run

1. `cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo` → SUCCESS
2. `cmake --build build/host` → SUCCESS (all targets)
3. `ctest --test-dir build/host --output-on-failure` → **150/150 PASSED**

---

## Build Results

| Mode | Configure | Build | CTest |
|------|-----------|-------|-------|
| host | PASS | PASS | **150/150 PASSED** |
| npu | PASS (configure) | PENDING | PENDING (requires ccec + ASC CMake integration) |

---

## Operator Status Table

| # | Operator | Category | Generated | Built (host) | Tests Passed | Accuracy | NPU Kernel | Notes |
|---|----------|----------|-----------|-------------|-------------|----------|------------|-------|
| 1 | elementwise_add | elementwise | yes | yes | 10/10 | pass | AscendC::Add (Vec) | Vector Core |
| 2 | axpy_scalar_affine | elementwise | yes | yes | 19/19 | pass | AscendC::Muls+Adds (Vec) | Scalar params |
| 3 | broadcast_binary | elementwise | yes | yes | 12/12 | pass | AscendC::Muls+Adds (Vec) | Broadcast |
| 4 | activation_gelu_silu | elementwise | yes | yes | 8/8 | pass | AscendC::Tanh+Exp+Div (Vec) | GELU/SiLU |
| 5 | reduce_sum_lastdim | reduction | yes | yes | 24/24 | pass | AscendC::ReduceSum (Vec) | Float32 accum |
| 6 | transpose_2d_4d | layout | yes | yes | 21/21 | pass | AscendC::DataCopy strided (MTE) | MTE2 bound |
| 7 | layer_norm | normalization | yes | yes | 15/15 | pass | AscendC::ReduceSum+Mul+Div (Vec) | Multi-pass |
| 8 | matmul_bias | matmul | yes | yes | 18/18 | pass | AscendC::MatMul (Cube) | Cube Core |
| 9 | rms_norm_rope_fused | fusion | yes | yes | 6/6 | pass | AscendC::ReduceSum+Mul+Sqrt (Vec) | LLM fusion |
| 10 | flash_attention_fwd_fused | attention | yes | yes | 17/17 | pass | AscendC::MatMul+Exp (Cube+Vec) | Online softmax |

**Total: 150/150 tests passing in host mode (100%)**

---

## Flash Attention Proof

- **Implementation status:** Complete — CPU golden with online softmax, AscendC kernel with MatMul + online softmax
- **Tiling strategy:** Tile-based along S_q and S_k dimensions, double-buffered, BlockIdx-based work distribution
- **Online softmax status:** Implemented in both golden (float64) and kernel (max-subtract, running sum, running accumulator)
- **Mask support:** Causal mask (query i attends keys j<=i), verified in tests
- **Golden correctness:** All 17 tests pass including causal mask constraint verification
- **Cube Core:** `AscendC::MatMul` for both Q@K^T and P@V GEMMs
- **NPU direct launch:** `extern "C" __global__ __aicore__ flash_attention_fwd_fused_kernel`
- **Blockers:** NPU compilation requires ccec + ASC CMake (`find_package(ASC)`)

---

## Future Arbitrary Operator Workflow Proof

| Step | Command | Status |
|------|---------|--------|
| Start new op | `/npu-new-op <<<...>>>` or `python3 scripts/new_op.py <name> <category>` | Working |
| Spec generation | `python3 scripts/generate_op_from_spec.py specs/<op>.yaml` | Working |
| State persistence | `.npu-harness/operators/<op>/{SPEC,PLAN,STATUS,DECISIONS,RESULTS,PERF,BLOCKERS}.{yaml,md}` | Working |
| Registry update | `OPERATOR_REGISTRY.yaml` auto-updated | Working |
| Build | `python3 scripts/harness.py build --mode host` | Working |
| Test | `python3 scripts/harness.py test --mode host --op <op>` | Working |
| Verify | `python3 scripts/harness.py verify <op>` | Working |

---

## Blockers

1. **NPU compilation:** Requires ASC CMake integration (`find_package(ASC)`, `.asc` file compilation via ccec)
2. **msprof profiling:** Requires compiled NPU kernel with AscendC compiler
3. **Ratio Bound verification:** Pending NPU execution — kernels designed for Vector/Cube/MTE bound (>0.8)

---

## Files Created

- CLAUDE.md, AGENTS.md, PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md, CONTEXT.md, README.md
- OPERATOR_REGISTRY.yaml, CMakeLists.txt, CMakePresets.json
- cmake/{Ascend,CompilerOptions,GTest,OperatorTarget}.cmake (4)
- include/harness/{accuracy,dtype,shape,tensor,logging,skip,ascend_runtime,test_utils}.hpp (8)
- src/{accuracy,dtype,shape,test_utils,ascend_runtime}.cpp (5)
- scripts/{harness,detect_ascend_env,generate_op_from_spec,new_op}.py + .sh (9)
- specs/01..10_*.yaml (10)
- ops/<op>/{CMakeLists,golden/*,host/*,runner/*,kernel/*,bench/*} × 10 (90)
- tests/test_*.cpp (11)
- docs/{architecture,build,direct_launch,testing,troubleshooting,operators/*}.md (15)
- .claude/commands/*.md (11)
- .claude/skills/*/SKILL.md (10)
- .npu-harness/{config,state/*,templates/*,runs/*} (12)

**Total: ~200 files**

---

## Next Commands for the User

```bash
# Check status
python3 scripts/harness.py status

# Host build & test (verified 150/150)
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host && ctest --test-dir build/host --output-on-failure

# NPU build (requires ccec + ASC CMake)
# - Add find_package(ASC) to cmake/Ascend.cmake
# - Use .asc extension for kernels or configure ASC language
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu && ctest --test-dir build/npu --output-on-failure

# Profile with msprof (after NPU build)
msprof --application=./build/npu/tests/test_elementwise_add --output=prof_data
```
