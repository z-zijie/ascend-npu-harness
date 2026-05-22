# ascend-c-kernel-implementation

## Skill Name

`ascend-c-kernel-implementation` -- Ascend C Kernel Implementation

## Trigger Description

The user requests implementation of an Ascend C NPU kernel. This typically follows `npu-spec-authoring` and `npu-tiling-design`. The task is to write the device-side kernel code (`kernel/*_kernel.cpp`), host-side tiling logic (`host/*_tiling.hpp`, `host/*_tiling.cpp`), and runner (`runner/*_runner.hpp`, `runner/*_runner.cpp`) that launches the kernel via direct `<<<>>>` syntax.

## When to Use

- A SPEC.yaml exists and the operator has `implementation.ascend_kernel: required`.
- The user says "implement the kernel for <op>" or "write the Ascend C kernel".
- `/npu-execute-op <op_name>` has reached the kernel implementation step.
- An existing kernel needs correction (out-of-bounds fix, dtype conversion fix, tiling parameter fix).
- A new dtype or shape variant needs kernel support.
- A performance tuning iteration requires kernel changes.

## When Not to Use

- No SPEC.yaml exists -- run `npu-spec-authoring` first.
- No tiling design exists -- run `npu-tiling-design` first.
- The task is to write CPU golden -- use `cpu-golden-authoring`.
- The task is to write tests -- use `gtest-verification`.
- The task is benchmarking -- use `performance-benchmarking`.
- CANN is unavailable and you are asked to implement the kernel anyway -- the kernel source can be written and checked for syntax, but it cannot be compiled. Write the code, guard it with preprocessor conditionals, and skip NPU compilation.

## Required Inputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` -- the authoritative spec for shapes, dtypes, tiling strategy, and constraints.
2. **Tiling design document** or the tiling decisions recorded in `npu-tiling-design` output (tile sizes, block dimensions, workspace requirements, tail policy).
3. **INTAKE.yaml** at `.npu-harness/operators/<op_name>/INTAKE.yaml` -- for math definition and parameter details.
4. **CANN toolchain** -- for compilation. If unavailable, the skill must produce correct source code that can compile when CANN is present, guarded by `#ifdef` for host-only builds.

## Required Outputs

1. **`ops/<op_name>/kernel/<op_name>_kernel.cpp`** -- the Ascend C device kernel.
2. **`ops/<op_name>/host/<op_name>_tiling.hpp`** -- tiling parameter struct and host-side declarations.
3. **`ops/<op_name>/host/<op_name>_tiling.cpp`** -- tiling computation and validation.
4. **`ops/<op_name>/runner/<op_name>_runner.hpp`** -- runner declarations.
5. **`ops/<op_name>/runner/<op_name>_runner.cpp`** -- host-side launch with `<<<>>>` and device memory management.
6. **`ops/<op_name>/CMakeLists.txt`** -- build integration for the operator (created or updated).

## Mandatory Rules

These 10 rules are non-negotiable. Violating any of them is a defect.

### Rule 1: Direct `<<<>>>` Launch

The kernel MUST be launched using the Ascend C direct launch syntax:

```cpp
kernel_name<<<block_dim, l2ctrl, stream>>>(/* args */);
```

Do NOT call PyTorch, torch_npu, TensorFlow, ONNX Runtime, ACLNN, or any other framework operator as the real implementation. The kernel is a custom Ascend C function.

### Rule 2: No Framework Operator Substitution

The kernel implementation must be self-contained Ascend C code. Do not write a wrapper that delegates to a framework op. If a high-level Ascend C API (e.g., Matmul, Softmax) is available and appropriate, it may be used as a component within the custom kernel, but the kernel entry point must be a custom `__global__ __aicore__` function.

### Rule 3: Target Arch dav-2201

All kernel code, CMake target flags, and launch parameters must target `dav-2201`. The arch must be explicit:

```cmake
set(ASCEND_ARCH "dav-2201")
target_compile_options(<target> PRIVATE --target=${ASCEND_ARCH})
```

### Rule 4: Clear Tiling with Documented Tile Sizes

Every kernel must have explicit tiling parameters. Tile sizes must be documented as named constants at the top of the kernel file:

```cpp
// Tiling constants for dav-2201
// Vector width: 32 elements (fp16)
// UB capacity: ~192KB per core
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 16;
constexpr int LOOP_K = K / TILE_K;  // must handle remainder
```

Tile sizes must be justified by UB capacity, vector width, and memory bandwidth. A comment explaining why each tile size was chosen is required.

### Rule 5: Safe Tail Handling

Every kernel must handle dimensions that are NOT multiples of the tile size. The tail handling logic must:

- Compute `remainder = total_size % tile_size`.
- Process the remainder without reading or writing out-of-bounds global memory.
- Use a separate tail loop or conditional bounds check. Example pattern:

```cpp
const int total_work = shape_info.total_elements;
const int core_work = (total_work + num_cores - 1) / num_cores;
const int core_start = core_idx * core_work;
const int core_end = min(core_start + core_work, total_work);
const int core_size = core_end - core_start;

for (int i = 0; i < core_size; i += TILE_ELEMS) {
    int tile_elems = (i + TILE_ELEMS <= core_size) ? TILE_ELEMS : (core_size - i);
    // ... process tile_elems elements
}
```

### Rule 6: No Out-of-Bounds Global Memory Access

Every global memory load and store must be bounds-checked. Use the patterns:

- `SetUserWorkspace` for workspace allocation.
- `DataCopy` with verified source/destination ranges.
- Explicit size checks before `CopyIn` / `CopyOut` or equivalent data movement APIs.

### Rule 7: Documented Dtype Conversion

All conversions between fp16 and fp32 must be explicit and documented:

```cpp
// Accumulate in fp32 for numerical stability
float acc = 0.0f;

// Load fp16 input, convert to fp32
float val = static_cast<float>(input_fp16[idx]);

// Compute in fp32
acc += val * weight;

// Convert back to fp16 for output
output_fp16[idx] = static_cast<half>(acc);
```

For bf16 support, use equivalent explicit conversions.

### Rule 8: Graceful Skip When CANN Unavailable

Use preprocessor guards so the file can be compiled (or at least parsed) in host-only mode:

```cpp
#ifdef HARNESS_ENABLE_ASCEND
// Ascend C kernel implementation
extern "C" __global__ __aicore__
void my_kernel(/* ... */) {
    // ...
}
#else
// Stub: CANN not available, kernel cannot be compiled
// Host-only mode does not require this file to produce object code.
#endif  // HARNESS_ENABLE_ASCEND
```

The runner must also check at runtime:

```cpp
#ifdef HARNESS_ENABLE_ASCEND
    if (!harness::ascend::IsAscendRuntimeAvailable()) {
        // Host-only mode; skip NPU launch
        return;
    }
    // ... launch kernel
#endif
```

### Rule 9: Correct Kernel Function Attributes

The kernel function must use the correct Ascend C attributes:

```cpp
extern "C" __global__ __aicore__
void operator_name_kernel(
    __gm__ float* input_a,
    __gm__ float* input_b,
    __gm__ float* output,
    TilingData tiling_data
) {
    // Kernel body
}
```

Adjust types and address space qualifiers to match the local CANN version. Use `__gm__` for global memory pointers.

### Rule 10: Correct Host-Side Launch Parameters

The host-side launch must use `block_dim` and `stream`:

```cpp
// block_dim: {blocks_per_core, cores_used}, total blocks = blocks_per_core * cores_used
uint32_t block_dim = static_cast<uint32_t>(num_cores);  // one block per core
::ascend::runtime::Stream stream = device_context.CreateStream();

kernel_name<<<block_dim, nullptr, stream>>>(
    device_input_a, device_input_b, device_output, tiling_data
);

stream.Synchronize();
```

## Step-by-Step Procedure

### Step 1: Read the SPEC and Tiling Design

Read `specs/<op_name>.yaml` and the tiling design output. Confirm you understand:
- Input shapes, dtypes, and layouts.
- Output shapes, dtypes, and layouts.
- Tile sizes and block dimensions.
- Tail handling requirements.
- Workspace requirements.

### Step 2: Create or Update CMakeLists.txt

Write `ops/<op_name>/CMakeLists.txt` with:
- A library target for the kernel (compiled with Ascend C compiler when available).
- A library target for host tiling code.
- A library target for the runner.
- Preprocessor guard for `HARNESS_ENABLE_ASCEND`.
- Proper arch flags (`--target=dav-2201`, equivalent).

### Step 3: Define the Tiling Parameter Struct

In `host/<op_name>_tiling.hpp`, define a struct that carries all tiling parameters from host to device:

```cpp
struct TilingData {
    uint32_t total_elements;
    uint32_t tile_elements;
    uint32_t tail_elements;
    uint32_t num_cores;
    // Shape-specific fields
    uint32_t dim0, dim1, dim2, dim3;
    // Stride info
    uint32_t stride0, stride1, stride2;
    // Scalar parameters (if any)
    float alpha;
    float beta;
    float eps;
};
```

All fields should be `uint32_t` or `float` -- avoid types that may differ between host and device.

### Step 4: Implement Host Tiling Computation

In `host/<op_name>_tiling.cpp`, implement a function that:
- Reads input shapes.
- Computes tile counts for each dimension.
- Computes tail/remainder sizes.
- Packs everything into the `TilingData` struct.
- Validates that dimensions are within supported ranges.

### Step 5: Implement the Device Kernel

In `kernel/<op_name>_kernel.cpp`:

1. **Kernel entry:** Use `extern "C" __global__ __aicore__` function signature.
2. **Unpack tiling data:** Read tiling parameters from the `TilingData` argument.
3. **Set workspace:** If workspace is needed, use `SetUserWorkspace`.
4. **Main loop:** Process data in tiles. For each tile:
   - Copy tile from global memory to UB (Unified Buffer) using `DataCopy` or equivalent.
   - Compute in UB.
   - Copy result from UB back to global memory.
5. **Tail loop:** Handle the remainder separately with correct bounds.
6. **Dtype conversions:** Explicit fp16 <-> fp32 conversions as documented.

### Step 6: Implement the Runner

In `runner/<op_name>_runner.hpp` and `runner/<op_name>_runner.cpp`:

1. **Input validation:** Check shapes, dtypes, parameter validity.
2. **Device memory allocation:** Allocate GM for all input/output tensors.
3. **Host-to-device copy:** Copy input data to device.
4. **Tiling computation:** Call the host tiling function.
5. **Kernel launch:** Use `<<<block_dim, nullptr, stream>>>`.
6. **Synchronize:** Wait for kernel completion.
7. **Device-to-host copy:** Copy output data back to host.
8. **Cleanup:** Free device memory, destroy streams (use RAII wrappers).

### Step 7: Verify with Host-Only Build

Run the host-only build to verify that preprocessor guards work:

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF
cmake --build build/host
```

The build must succeed. The kernel file should be either excluded or compiled as an empty translation unit.

### Step 8: Verify with NPU Build (if CANN available)

```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201
cmake --build build/npu
```

Fix compilation errors before proceeding to tests.

## Verification Checklist

Before declaring kernel implementation complete, verify each item:

- [ ] Kernel function uses `extern "C" __global__ __aicore__` attributes.
- [ ] Kernel is launched via `<<<block_dim, nullptr, stream>>>` in the runner.
- [ ] No framework operator substitution anywhere in the launch path.
- [ ] Target arch is `dav-2201` (check CMake flags and kernel build).
- [ ] Tile sizes are documented as named constants with justification comments.
- [ ] Tail handling covers all dimensions that may not be multiples of tile size.
- [ ] No out-of-bounds global memory access for any valid shape.
- [ ] Dtype conversions (fp16 <-> fp32) are explicit and documented.
- [ ] Preprocessor guards (`#ifdef HARNESS_ENABLE_ASCEND`) protect Ascend C code.
- [ ] Host-only build succeeds without CANN.
- [ ] `block_dim` is correctly computed (one block per core, or documented multi-block layout).
- [ ] Stream is created and synchronized correctly.
- [ ] Device memory is allocated and freed (RAII preferred).
- [ ] TilingData struct uses host/device-compatible types (uint32_t, float).
- [ ] Host tiling validates input dimensions before launching.
- [ ] Runner handles missing NPU gracefully at runtime (GTEST_SKIP equivalent for benchmarks).
- [ ] CMakeLists.txt for the operator is complete and integrates with parent build.
- [ ] All generated files are listed in the harness run log.
