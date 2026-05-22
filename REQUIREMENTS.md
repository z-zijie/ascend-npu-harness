# REQUIREMENTS.md — Harness Requirements

## Functional Requirements

### FR1: Operator Intake
The Harness must accept natural-language operator requirements and convert them into structured SPEC.yaml files.

### FR2: Spec-Driven Development
Every operator must have a SPEC.yaml before implementation begins. No spec, no code.

### FR3: CPU Golden Reference
Every operator must have a C++ CPU reference implementation for correctness verification.

### FR4: GTest/CTest Integration
Every operator must have GTest tests registered with CTest labels.

### FR5: Host-Only Mode
The project must build and test without CANN or NPU hardware.

### FR6: NPU Mode
When CANN and NPU hardware are available, Ascend C kernels must be built and tested.

### FR7: Direct Launch
NPU kernels must use `<<<>>>` direct launch, not framework operator substitution.

### FR8: Performance Contract
Operators with performance requirements must be benchmarked and compared to targets.

### FR9: State Persistence
Harness state must survive context resets via filesystem state files.

### FR10: Operator Registry
OPERATOR_REGISTRY.yaml must be the source of truth for operator status.

## Non-Functional Requirements

### NFR1: Deterministic Tests
All randomized tests must use deterministic seeds, printed on failure.

### NFR2: Graceful Degradation
Missing CANN/NPU must result in clear skip messages, not build failures.

### NFR3: Precise Blockers
Incomplete work must produce specific blocker reports (missing header, library, API, device).

### NFR4: Target Architecture
Default NPU architecture is dav-2201. No other default.

## Operator Requirements

Each operator must provide:
1. SPEC.yaml with complete schema
2. CPU golden implementation
3. Ascend C kernel (when NPU available)
4. Host tiling/shape inference
5. `<<<>>>` launch path
6. GTest tests with CTest labels
7. Operator documentation
8. Accuracy tolerances per dtype
9. Performance benchmark (when applicable)
