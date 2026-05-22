# npu-tiling-design

## Skill Name

`npu-tiling-design` -- NPU Tiling and Scheduling Design

## Trigger Description

The user requests tiling and scheduling design for an NPU operator. This happens after spec authoring and before kernel implementation. The task is to determine tile sizes, block dimensions, alignment constraints, workspace requirements, tail policy, per-core work split, memory movement plan, and performance risks.

## When to Use

- A SPEC.yaml exists and the operator has `implementation.ascend_kernel: required`.
- The user says "design the tiling for <op>" or "how should I tile <op>?".
- An existing tiling design needs revision due to performance gaps or correctness issues.
- The operator is a matmul, attention, reduction, or fusion -- any operator where tiling is non-trivial.
- The user asks about UB capacity limits, vector width alignment, or core utilization for a specific operator.

## When Not to Use

- No SPEC.yaml exists -- run `npu-spec-authoring` first.
- The operator is a trivial elementwise op with auto-tiling -- document that the default per-element tiling is sufficient and move on.
- The user is debugging a kernel -- use `systematic-debugging`.
- The user is benchmarking -- use `performance-benchmarking`.
- The task is to write tests -- use `gtest-verification`.

## Required Inputs

1. **SPEC.yaml** at `specs/<op_name>.yaml` -- provides shapes, dtypes, tiling strategy, and constraints.
2. **NPU arch constraints** for `dav-2201`:
   - Vector width: 32 elements (fp16) or 16 elements (fp32) per vector operation.
   - UB capacity: approximately 192 KB per AI Core.
   - L1 buffer: architecture-specific, typically ~in the same range as UB.
   - Maximum block dimension: 65535 blocks.
   - Alignment: 32B or 64B alignment for efficient global memory access.
   - GM bandwidth: ~1.2 TB/s peak (architecture-dependent, use as planning assumption).
3. **Input shape ranges** from SPEC.yaml `shape_symbols` and `tests.shapes`.

## Required Outputs

A tiling design document or set of decisions covering all eight mandatory sections below. The output should be written to `.npu-harness/operators/<op_name>/TILING.md` or integrated into PLAN.md.

### 1. Tile Sizes

For each dimension being tiled, specify:
- Full dimension range.
- Tile size in elements.
- Justification (UB capacity, vector width alignment, memory coalescing).

Example:
```
Dimension M: range [1, 4096], tile M_TILE = 128
  Justification: 128 fp16 elements = 256 bytes, fits UB. Aligned to 32-wide vector.
Dimension N: range [1, 4096], tile N_TILE = 128
  Justification: Same as M.
Dimension K: range [1, 4096], tile K_TILE = 16
  Justification: Inner reduction, keep in registers to avoid UB spilling. 16 is multiple of vector width for fp32 accumulation.
```

Tile sizes MUST be aligned to the NPU vector width. For dav-2201:
- fp16: tile size must be a multiple of 32.
- fp32: tile size must be a multiple of 16.
- int8/int16: consult the architecture manual for vector lane width.

### 2. Block Dimensions

Specify the block layout:

```
Total blocks = num_cores * blocks_per_core
block_dim = num_cores  (one block per core, each core processes its partition)
```

The total blocks must not exceed 65535. For most operators, one block per core is the correct configuration. Multi-block-per-core is only needed when:
- Each core must process >1 independent tile stream.
- Vector pipelining requires tile-level parallelism within a core.
- The operator has independent sub-computations that map naturally to multiple blocks.

### 3. Alignment Constraints

List all alignment requirements:

```
1. Global memory base addresses: 32B or 64B aligned.
2. Tile dimensions: multiples of vector width.
3. UB buffer base addresses: 32B aligned.
4. DMA copy sizes: multiples of 32B for efficient transfer.
5. Workspace buffers: 32B aligned.
```

Any violation of alignment must be called out as a performance risk or correctness hazard.

### 4. Workspace Requirements

Quantify the per-core workspace needed:

```
UB workspace per core:
  - Input tile A:  M_TILE * K_TILE * sizeof(fp16) = 128*16*2 = 4096 bytes
  - Input tile B:  K_TILE * N_TILE * sizeof(fp16) = 16*128*2 = 4096 bytes
  - Output tile C: M_TILE * N_TILE * sizeof(fp32) = 128*128*4 = 65536 bytes
  - Scratch/accumulator: varies by operator
  Total UB: <total> bytes, must be <= 192KB (~196608 bytes)
```

Workspace must be bounded and documented. If workspace exceeds UB capacity, redesign tiling to use smaller tiles or multi-pass strategies.

### 5. Tail Policy

Explicitly describe how remainder elements are handled for each tiled dimension:

```
Policy: per-dimension tail tile processing

For each tiled dimension D with size D_total and tile size D_TILE:
  num_full_tiles = D_total / D_TILE
  remainder       = D_total % D_TILE

  Full tiles (0 .. num_full_tiles - 1): use D_TILE elements.
  Tail tile (if remainder > 0): use 'remainder' elements.
    - GM copy size is 'remainder * element_size' bytes.
    - Computation is bounded to 'remainder' elements.
    - No out-of-bounds access: loop limits are min(idx + D_TILE, D_total).

Multi-dimension tail handling:
  When multiple dimensions have tails simultaneously,
  all combinations of {full, tail} must be handled.
  The tail is the last tile in each tiled dimension.
```

### 6. Per-Core Work Split

Describe how work is distributed across cores:

```
Strategy: split along outermost tiled dimension

For a tensor of shape [B, M, N] with tiling along M and N:
  - Split M into M_per_core = ceil(M_total / num_cores).
  - Each core processes rows [core_id * M_per_core, min((core_id+1) * M_per_core, M_total)).
  - Within each core's M partition, iterate over N in N_TILE-sized chunks.

Load balance analysis:
  - Best case: M_total % num_cores == 0 (perfect balance).
  - Worst case: M_total = 1, num_cores = 32 (31 cores idle).
    -> Mitigation: if M_total < num_cores, use fewer cores or split along a different dimension.
  - Typical case: M_total >> num_cores, imbalance is < 1 tile per core (acceptable).
```

### 7. Memory Movement Plan

Describe the complete data movement for one core:

```
For each tile iteration:
  1. GM -> UB: Load input tile A (M_TILE * K_TILE) via DMA.
  2. GM -> UB: Load input tile B (K_TILE * N_TILE) via DMA.
  3. Compute: A @ B in UB/registers -> partial C (M_TILE * N_TILE) in fp32.
  4. Accumulate: add to running C accumulator in UB.
  5. After all K tiles: optionally apply bias, activation function.
  6. UB -> GM: Store output tile C (M_TILE * N_TILE) with dtype conversion if needed.

Pipelining (where applicable):
  - Double-buffer: while computing tile(k), DMA-load tile(k+1).
  - Overlap GM->UB load of one tile with computation of the previous tile.
  - Requires 2x UB allocation for the pipelined buffer.
```

### 8. Performance Risks

Identify and rate potential performance issues:

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|------------|
| Bank conflicts | Medium | Access pattern A[B,M,K] with stride=1 on inner dim may hit same bank. | Pad inner dim to avoid conflict stride. |
| Pipeline bubbles | Low | DMA latency between K tiles if no double-buffering. | Use double-buffering for K loop. |
| Load imbalance | Low | Non-uniform M split when M_total is small. | Accept for now; tune if benchmark shows >10% imbalance. |
| UB spill | High | If workspace exceeds 192KB, data spills to L1, causing >10x slowdown. | Redesign tile sizes to fit UB. |
| GM bandwidth saturation | Medium | Large contiguous copies may saturate GM bus. | Use vectorized DMA; this is usually a good problem to have. |

## Step-by-Step Procedure

### Step 1: Analyze the SPEC.yaml

Read the spec. Extract:
- All shape symbols and their ranges.
- All dtypes (determines element sizes).
- `tiling.strategy` (auto vs manual).
- `tiling.constraints` (any alignment or size constraints).
- Whether performance targets exist.

### Step 2: Determine the Tiling Dimensions

Identify which dimensions should be tiled:

- **Elementwise ops:** tile along the flattened total elements. Tile size = max elements fitting UB.
- **Reduction ops:** tile along the non-reduced dimensions; the reduce dimension is iterated in an inner accumulation loop (K-like).
- **Matmul:** tile M, N, K. M_TILE * N_TILE * element_size + input tile buffers must fit UB.
- **Layout ops:** tile blocks of the source tensor for contiguous reads; write to destination with appropriate strides.
- **Normalization:** two-pass approach. Pass 1: partial reduce over the norm dimension. Pass 2: apply scale.
- **Attention (flash):** tile Q along S_q, tile KV along S_k (outer loop + inner loop). M_TILE = S_q tile, N_TILE = S_k tile. Separate tiles for Q tile and KV tiles. Maintain online softmax statistics.

### Step 3: Compute Tile Sizes

For each tiled dimension:

1. Start with the largest power-of-2 or multiple-of-32 that fits all buffer requirements in UB.
2. Check alignment: tile_size % vector_width == 0.
3. Verify UB budget: sum of all per-tile buffers <= 192KB for dav-2201.
4. If oversubscribed, reduce tile sizes and iterate.

### Step 4: Design the Core Work Split

Choose one of these strategies:
- **Split outermost dim:** best for matmul (split M), normalization (split batch or rows).
- **Split innermost dim:** useful for elementwise (flat split).
- **Multi-dim split:** advanced, for large tensors where a single dimension is too small.

Always compute the maximum imbalance: `max_work_per_core / min_work_per_core`. If >2.0, document as a risk.

### Step 5: Design the Memory Movement Plan

For each data type used in the operator:

1. GM -> UB: contiguous read of tile-sized block. Use DMA copy. Prefer 32B-aligned sizes.
2. Compute: in UB, using vector operations.
3. UB -> GM: contiguous write of result tile. Use DMA copy.

If double-buffering or pipelining is applicable:
- Allocate 2x the tile buffer for the pipelined dimension.
- Document the pipeline stages.

### Step 6: Design Tail Handling

For each tiled dimension:
- Compute `num_full = total / tile_size`.
- Compute `remainder = total % tile_size`.
- Write the logic for when `remainder > 0`.

Multi-dim case: when more than one dimension has a tail in the same tile, all combinations must be handled. The code pattern should iterate over dimensions independently and apply min(tile_size, remaining) for each.

### Step 7: Quantify Workspace

List every buffer in UB:
- Name, size in elements, element size in bytes, total bytes.
- Whether it is single-buffered or double-buffered.
- Whether it persists across tile iterations or is transient.

Sum all buffers. Verify sum <= 192KB.

### Step 8: Identify and Rate Risks

Use the risk categories from the template. For each risk:
- Name it.
- Rate severity (Low / Medium / High / Blocker).
- Explain the mechanism.
- Propose a mitigation or accept with justification.

### Step 9: Write the Tiling Design Output

Write the design to `.npu-harness/operators/<op_name>/TILING.md`. The document must include all eight sections. Use tables and concrete numbers, not generic prose.

### Step 10: Validate Against Arch Limits

Run the following sanity checks:
- Any tile size not a multiple of vector width -> redesign.
- Total UB > 192KB -> redesign.
- Total blocks > 65535 -> redesign.
- Per-core work imbalance > 2x and performance required -> add risk note, consider redistribution.

## Verification Checklist

Before declaring tiling design complete, verify each item:

- [ ] Every tiled dimension has a documented tile size with justification.
- [ ] Every tile size is aligned to dav-2201 vector width (32 for fp16, 16 for fp32).
- [ ] Block dimensions are specified and within the 65535 limit.
- [ ] Alignment constraints are documented (32B or 64B for GM, 32B for UB).
- [ ] Workspace requirements are quantified in bytes, with per-buffer breakdown.
- [ ] Total workspace fits within UB capacity (~192KB for dav-2201).
- [ ] Tail policy handles all remainder cases for all tiled dimensions.
- [ ] Multi-dimension tail combinations are accounted for.
- [ ] Per-core work split strategy is documented with load balance analysis.
- [ ] Memory movement plan covers all input and output tensors (GM->UB, UB->GM).
- [ ] Pipelining or double-buffering is documented where applicable.
- [ ] Performance risks are listed with severity ratings and mitigations.
- [ ] TILING.md (or equivalent) is written and contains all eight sections.
- [ ] All arch limits (vector width, UB capacity, block count) are explicitly verified.
