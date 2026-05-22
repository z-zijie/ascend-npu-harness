# cpu-golden-authoring

## Skill Name

`cpu-golden-authoring` -- CPU Golden Reference Implementation

## Trigger Description

The user requests a CPU golden reference implementation for an NPU operator. The CPU golden is the ground-truth computation against which the NPU kernel output is compared. It must be correct, deterministic, numerically stable, and free of framework dependencies.

## When to Use

- A SPEC.yaml exists and the operator has `implementation.cpu_golden: required`.
- The user says "write the CPU golden for <op>" or "implement the reference for <op>".
- `/npu-execute-op <op_name>` has reached the CPU golden step.
- An accuracy comparison reports mismatches and the golden needs to be verified or corrected.
- A new dtype, shape, or parameter variant needs golden support.

## When Not to Use

- No SPEC.yaml exists -- run `npu-spec-authoring` first.
- The task is to write the NPU kernel -- use `ascend-c-kernel-implementation`.
- The task is to write tests (though tests will call the golden) -- use `gtest-verification`.
- An existing golden is correct and only the NPU kernel needs debugging -- use `systematic-debugging`.
- The user asks for a PyTorch reference -- this is not the CPU golden skill. Explain that the harness requires a C++ golden and PyTorch may be used as an optional cross-check only.

## Required Inputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` -- provides math definition, shapes, dtypes, parameters, and accuracy tolerances.
2. **INTAKE.yaml** at `.npu-harness/operators/<op_name>/INTAKE.yaml` -- provides the math formula in prose form.
3. **Harness accuracy utilities** at `include/harness/accuracy.hpp` -- for standard comparison when testing.

## Required Outputs

1. **`ops/<op_name>/golden/<op_name>_golden.hpp`** -- header declaring the golden function(s).
2. **`ops/<op_name>/golden/<op_name>_golden.cpp`** -- implementation of the golden function(s).

The golden must be:
- Callable from GTest without pulling in any DL framework.
- Header-only OR separately compiled (both patterns are supported).
- Namespaced under `harness::golden::<op_name>` (recommended).

## Mandatory Requirements

### Requirement 1: Pure C++ Implementation

The golden must be implemented in standard C++17. No PyTorch, no TensorFlow, no NumPy, no Python bindings, no OpenBLAS, no MKL. Allowed dependencies:
- Standard C++ library (`<cmath>`, `<vector>`, `<algorithm>`, `<numeric>`, `<cstring>`, etc.).
- Harness utility headers (`harness/dtype.hpp`, `harness/shape.hpp`, `harness/tensor.hpp`).
- Nothing else.

If the operator requires a computation that is impractical in pure C++ (e.g., a full BLAS GEMM for large matrices), the golden may use a naive triple-loop implementation. The golden prioritizes correctness over performance. A slow but correct golden is acceptable.

### Requirement 2: Deterministic for Given Inputs

The golden must produce bit-exact identical output for the same input values, shapes, and parameters. No random seeds, no non-deterministic algorithms. If the operator spec requires randomization (e.g., dropout), the golden must accept a seed parameter and use a deterministic PRNG (e.g., `std::mt19937`).

### Requirement 3: Float32 Accumulation for Reductions

When accumulating over a large number of elements, the golden must use float32 (or float64 for very large reductions) regardless of input dtype. Pattern:

```cpp
// Correct: accumulate in fp32
float accumulator = 0.0f;
for (int k = 0; k < K; ++k) {
    accumulator += static_cast<float>(input[row * K + k]);
}
output[row] = accumulator;

// Also correct for very large K: use double
double accumulator = 0.0;
for (int k = 0; k < K; ++k) {
    accumulator += static_cast<double>(input[row * K + k]);
}
```

### Requirement 4: Numerically Stable Softmax

If the operator includes softmax (e.g., flash attention), use the max-subtract pattern:

```cpp
// Stable softmax for a vector of length N
float max_val = -std::numeric_limits<float>::infinity();
for (size_t i = 0; i < N; ++i) {
    max_val = std::max(max_val, static_cast<float>(scores[i]));
}
float sum_exp = 0.0f;
for (size_t i = 0; i < N; ++i) {
    sum_exp += std::exp(static_cast<float>(scores[i]) - max_val);
}
for (size_t i = 0; i < N; ++i) {
    probs[i] = std::exp(static_cast<float>(scores[i]) - max_val) / sum_exp;
}
```

### Requirement 5: Numerically Stable Reduction

For reductions over large dimensions (K > 1024), use techniques that reduce floating-point error:

- **Pairwise summation** (divide-and-conquer summation) for K up to ~10^5.
- **Kahan compensated summation** for K up to ~10^6.
- **Double-double** or `long double` for K > 10^6.

The choice must be documented in a comment. The default is simple float32 accumulation unless the spec or dimension sizes warrant a more stable technique.

```cpp
// Kahan summation for large reductions
float sum = 0.0f;
float compensation = 0.0f;
for (int k = 0; k < K; ++k) {
    float y = static_cast<float>(input[row * K + k]) - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
}
output[row] = sum;
```

### Requirement 6: No Reliance on PyTorch or TensorFlow

PyTorch, TensorFlow, or any other DL framework must not be a required dependency of the golden. If the user or harness wants a framework-based cross-check, it must be a separate, optional test (`#ifdef HAS_TORCH`) that does not block host-only builds.

### Requirement 7: Full Shape/Dtype Coverage

The golden must handle all shapes listed in `specs/<op_name>.yaml` under `tests.shapes` and any shape derivable from `shape_symbols`. It must also handle all dtypes listed in `inputs[].dtype` and `outputs[].dtype`.

Template for multi-dtype handling:

```cpp
template <typename T>
void GoldenImpl(const T* input, T* output, size_t count, float alpha, float beta) {
    for (size_t i = 0; i < count; ++i) {
        float val = static_cast<float>(input[i]);
        float result = alpha * val + beta;
        output[i] = static_cast<T>(result);
    }
}

// Explicit instantiations
template void GoldenImpl<float>(const float*, float*, size_t, float, float);
template void GoldenImpl<half_float::half>(const half_float::half*, half_float::half*, size_t, float, float);
```

### Requirement 8: Header-Only or Separately Compiled

Both patterns are acceptable:
- **Header-only:** template implementation in the `.hpp` file. Include directly in tests.
- **Separately compiled:** declaration in `.hpp`, implementation in `.cpp`. Link via CMake.

For header-only, add include guards and namespace:

```cpp
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace harness::golden::<op_name> {
    // ... functions
}  // namespace harness::golden::<op_name>
```

## Step-by-Step Procedure

### Step 1: Understand the Math

Read the `math.definition` from SPEC.yaml and INTAKE.yaml. Write down the exact formula in a comment at the top of the golden header. This serves as the contract documentation.

### Step 2: Choose the Numeric Kernel

For each supported dtype, decide:
- Input precision: the dtype of input data.
- Accumulation precision: fp32, fp64, or double.
- Output precision: the dtype of output data.
- Special handling needed: max-subtract, Kahan, pairwise.

Write these decisions as comments in the golden source.

### Step 3: Implement the Core Function

Write one or more functions that compute the operator. Keep the interface clean:

```cpp
// Minimally: input pointer, output pointer, shape info, parameters
template <typename T>
void Compute(
    const T* x,         // input tensor (contiguous)
    T* y,               // output tensor (contiguous)
    const int64_t* dims, // shape array
    int ndim,            // number of dimensions
    float alpha,         // scalar parameter
    float beta           // scalar parameter
);
```

### Step 4: Add Shape and Parameter Validation

Add lightweight input validation at the top of the golden function:

```cpp
if (x == nullptr || y == nullptr) {
    throw std::invalid_argument("Null input/output pointer in <op_name> golden");
}
if (ndim <= 0 || ndim > HARNESS_MAX_DIMS) {
    throw std::invalid_argument("Invalid number of dimensions in <op_name> golden");
}
// ... additional checks
```

### Step 5: Handle Edge Cases

Implement correct behavior for:
- Zero-element tensors (sum = 0, product = 1, etc.).
- Singleton dimensions (dim size = 1).
- `nan` and `inf` propagation (document the expected behavior).
- Very large values near dtype limits.
- Subnormal numbers.

### Step 6: Verify Numeric Stability

Walk through the computation mentally or with a small test case:
- Does accumulation use sufficient precision?
- Does softmax use max-subtract?
- Does division avoid division-by-zero (e.g., `eps` addition)?

### Step 7: Write the Header and Implementation

- Header: function declarations, template implementations if header-only, namespace, include guard.
- Implementation: function bodies, explicit template instantiations if needed.

### Step 8: Validate Against Simple Test Cases

Before handing off to `gtest-verification`, manually verify with a few known cases:
- All zeros input -> expected output.
- All ones input -> expected output.
- A 2x2 or 1x4 case computed by hand.

## Verification Checklist

Before declaring the golden complete, verify each item:

- [ ] Golden is implemented in pure C++ (C++17 standard library only + harness utilities).
- [ ] Golden is deterministic -- same inputs produce exactly the same outputs.
- [ ] Float32 (or float64) accumulation is used for inner reduction loops.
- [ ] Softmax uses max-subtract pattern if applicable.
- [ ] Large reductions use Kahan or pairwise summation if K > 1024.
- [ ] No `#include` of PyTorch, TensorFlow, or any DL framework.
- [ ] Golden handles every shape listed in SPEC.yaml `tests.shapes`.
- [ ] Golden handles every dtype listed in SPEC.yaml `inputs[].dtype`.
- [ ] Golden handles edge cases: zero-size, singleton, nan, inf, subnormals.
- [ ] Division-by-zero and log-of-negative are prevented or documented.
- [ ] Function is namespaced (recommended: `harness::golden::<op_name>`).
- [ ] Header has include guard (`#pragma once` or equivalent).
- [ ] Code compiles with the host-only build (`HARNESS_ENABLE_ASCEND=OFF`).
- [ ] Manual spot-checks against hand-computed values pass.
- [ ] Comments document the math formula, accumulation precision, and any numerical stability techniques.
