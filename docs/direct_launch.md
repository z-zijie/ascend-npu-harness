# Direct Launch Model

## Overview

The Harness uses Ascend C direct kernel launch with `<<<>>>` syntax for all NPU operator execution.

This is NOT framework operator substitution. Each operator implements a real custom Ascend C kernel.

## Kernel Declaration

```cpp
extern "C" __global__ __aicore__
void operator_kernel(
    __gm__ float* input,
    __gm__ float* output,
    int64_t num_elements);
```

## Host-Side Launch

```cpp
// Set up block dimensions
uint32_t block_dim = 32;  // threads per block
// Launch
operator_kernel<<<num_blocks, l2ctrl, stream>>>(
    device_input_ptr,
    device_output_ptr,
    num_elements
);
```

## Key Constraints

1. Target arch is always **dav-2201** unless explicitly overridden
2. All global memory pointers must be device-allocated
3. Tiling parameters are passed as kernel arguments
4. Block dimensions must respect NPU arch limits
5. Tail handling is required for non-multiple-of-tile sizes

## Graceful Degradation

When CANN or NPU is unavailable:
- Host-only build: kernels are not compiled
- NPU tests: use `GTEST_SKIP()` with reason message
- Runners: fall back to CPU golden

```cpp
#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
// Kernel code here
#endif
```

## Local CANN Adaptation

The exact kernel attributes and pointer types may vary with CANN version.
The `scripts/detect_ascend_env.py` script detects the local environment.
Refer to the local CANN documentation for version-specific details.

## Expected CANN 9.0.0 Conventions

- Kernel source extension: `.cpp` (compiled by ccec)
- Global memory: `__gm__` address space
- Block index: `get_block_idx_x()`
- Thread index: `get_thread_idx_x()`
- Stream: `aclrtStream` (from AscendCL)
