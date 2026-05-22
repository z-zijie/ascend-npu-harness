# Testing

## Test Framework

- GTest for C++ unit tests
- CTest for test discovery and execution
- Deterministic seeds for reproducibility

## CTest Labels

| Label | Description |
|-------|-------------|
| host | Host-only tests (always run) |
| golden | CPU golden comparison tests |
| npu | NPU integration tests |
| elementwise | Elementwise operators |
| reduction | Reduction operators |
| layout | Layout transform operators |
| normalization | Normalization operators |
| matmul | Matrix multiplication |
| attention | Attention operators |
| fusion | Fused operators |
| llm | LLM-related operators |
| slow | Long-running tests |
| benchmark | Performance benchmarks |

## Running Tests

```bash
# All tests
ctest --test-dir build/host --output-on-failure

# By label
ctest --test-dir build/host -L golden --output-on-failure
ctest --test-dir build/host -L elementwise --output-on-failure

# By name
ctest --test-dir build/host -R elementwise_add --output-on-failure
ctest --test-dir build/host -R flash_attention --output-on-failure
```

## NPU Test Behavior

NPU tests use graceful skip when hardware is unavailable:

```cpp
if (!harness::IsAscendRuntimeAvailable()) {
    GTEST_SKIP() << "Ascend runtime not available";
}
if (!harness::HasUsableDevice()) {
    GTEST_SKIP() << "No usable NPU device";
}
```

## Tolerance Guidelines

| Operator Type | float32 abs | float32 rel | float16 abs | float16 rel |
|--------------|-------------|-------------|-------------|-------------|
| Elementwise | 1e-5 | 1e-5 | 1e-2 | 1e-2 |
| Reduction | 1e-4 | 1e-4 | 5e-2 | 5e-2 |
| Matmul | 1e-4 | 1e-4 | 1e-1 | 1e-1 |
| Attention | 1e-4 | 1e-4 | 1e-1 | 1e-1 |

## Deterministic Seeds

All randomized tests use seed `20260522`. The seed is printed on failure for reproducibility.
