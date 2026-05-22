# Template: Operator Implementation Plan

## Operator: {OP_NAME}

## Overview
- **Category:** {CATEGORY}
- **Target Arch:** dav-2201
- **Spec:** specs/{OP_NAME}.yaml

## Implementation Steps

### Step 1: CPU Golden
- [ ] Write `ops/{OP_NAME}/golden/{OP_NAME}_golden.cpp`
- [ ] Verify against mathematical definition
- [ ] Test edge cases manually

### Step 2: Shape Validation
- [ ] Implement shape checking in host tiling
- [ ] Reject unsupported shapes with clear messages

### Step 3: Tiling Design
- [ ] Determine tile sizes based on data types and shapes
- [ ] Document block dimensions
- [ ] Handle tail cases

### Step 4: Ascend C Kernel
- [ ] Write `ops/{OP_NAME}/kernel/{OP_NAME}_kernel.cpp`
- [ ] Target arch: dav-2201
- [ ] Use `<<<>>>` direct launch

### Step 5: Runner Integration
- [ ] Write `ops/{OP_NAME}/runner/{OP_NAME}_runner.cpp`
- [ ] NPU path: kernel launch
- [ ] CPU fallback: call golden

### Step 6: Tests
- [ ] Write `tests/test_{OP_NAME}.cpp`
- [ ] Shape coverage from spec
- [ ] Dtype coverage from spec
- [ ] Edge cases
- [ ] Golden comparison

### Step 7: Build and Verify
- [ ] `cmake --build build/host`
- [ ] `ctest --test-dir build/host -R {OP_NAME}`
- [ ] Fix any failures

### Step 8: NPU Verification (if available)
- [ ] `cmake --build build/npu`
- [ ] `ctest --test-dir build/npu -R {OP_NAME}`

### Step 9: Benchmark (if required)
- [ ] Run benchmark
- [ ] Compare to performance targets
- [ ] Tune if needed

## Files
- `ops/{OP_NAME}/golden/{OP_NAME}_golden.hpp`
- `ops/{OP_NAME}/golden/{OP_NAME}_golden.cpp`
- `ops/{OP_NAME}/host/{OP_NAME}_tiling.hpp`
- `ops/{OP_NAME}/host/{OP_NAME}_tiling.cpp`
- `ops/{OP_NAME}/kernel/{OP_NAME}_kernel.cpp`
- `ops/{OP_NAME}/runner/{OP_NAME}_runner.hpp`
- `ops/{OP_NAME}/runner/{OP_NAME}_runner.cpp`
- `ops/{OP_NAME}/bench/bench_{OP_NAME}.cpp`
- `ops/{OP_NAME}/CMakeLists.txt`
- `tests/test_{OP_NAME}.cpp`
- `docs/operators/{OP_NAME}.md`

## Verification Commands
```bash
cmake --build build/host
ctest --test-dir build/host -R {OP_NAME} --output-on-failure
```

## Blockers
None identified.
