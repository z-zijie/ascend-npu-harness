# AGENTS.md — Harness Agent Roles

This document defines the agent roles used by the NPU Operator Development Harness. Each role represents a distinct responsibility boundary in the operator development workflow.

## Role 1: Operator Requirement Analyst

**Responsibility:** Convert natural-language operator descriptions into structured intake forms.
**Inputs:** User description, existing operator registry, Harness constraints.
**Outputs:** Structured intake YAML (name, category, math definition, inputs/outputs, shape rules, dtype rules, accuracy contract, performance contract).
**Failure modes:** Missing shape semantics, ambiguous dtype behavior, unspecified broadcasting, unclear reduction axes, missing performance targets when performance matters.
**Verification checklist:**
- [ ] Operator name is unique in registry
- [ ] All inputs have named, shaped, typed definitions
- [ ] All outputs have explicit shape derivation rules
- [ ] Dtype rules specify accumulation width
- [ ] Accuracy contract has explicit tolerances per dtype
- [ ] Performance contract is present or explicitly marked N/A

## Role 2: Spec Author

**Responsibility:** Produce valid SPEC.yaml from intake, enforce schema completeness.
**Inputs:** Intake YAML, operator category knowledge, schema definition.
**Outputs:** SPEC.yaml, updated OPERATOR_REGISTRY.yaml.
**Failure modes:** Missing shape constraints, missing tolerance, undefined edge cases, ambiguous layout rules, missing test shape coverage.
**Verification checklist:**
- [ ] Schema version is set
- [ ] Name, category, target_arch are present
- [ ] All inputs have dtype, shape, layout, role
- [ ] All outputs have dtype, shape, layout
- [ ] Shape symbols have allowed value ranges
- [ ] Shape rules are explicit
- [ ] Accuracy tolerances per dtype are set
- [ ] Performance targets or explicit "not required" marker
- [ ] Test shapes cover edge cases
- [ ] Status fields initialized to false

## Role 3: Tiling Designer

**Responsibility:** Design tiling strategy for Ascend C kernel execution.
**Inputs:** SPEC.yaml, NPU arch constraints (dav-2201), data sizes.
**Outputs:** Tile sizes, block dimensions, alignment constraints, workspace requirements, tail policy, per-core work split, memory movement plan.
**Failure modes:** Unaligned tile boundaries, excessive workspace, tail-handling gaps, imbalanced core utilization, bank conflicts.
**Verification checklist:**
- [ ] Tile sizes aligned to NPU vector width
- [ ] Block dimensions within arch limits
- [ ] Tail policy handles all shape remainders
- [ ] Workspace size is bounded and documented
- [ ] Memory movement plan covers all input/output tensors
- [ ] Per-core work split is balanced for common shapes

## Role 4: Ascend C Kernel Engineer

**Responsibility:** Implement Ascend C kernel with direct `<<<>>>` launch.
**Inputs:** SPEC.yaml, tiling design, NPU arch constraints.
**Outputs:** kernel/*_kernel.cpp, host/*_tiling.hpp, host/*_tiling.cpp.
**Failure modes:** Out-of-bounds access, incorrect dtype conversion, missing tail handling, wrong block dimensions, framework substitution, wrong arch target.
**Verification checklist:**
- [ ] Uses `<<<>>>` direct launch, not framework op
- [ ] Target arch is dav-2201
- [ ] Tiling parameters are passed correctly
- [ ] Tail handling covers all edge cases
- [ ] No out-of-bounds global memory access
- [ ] Dtype conversions are explicit and correct
- [ ] Graceful skip when CANN unavailable (host-only mode)

## Role 5: C++ Golden Engineer

**Responsibility:** Implement CPU reference implementation for correctness verification.
**Inputs:** SPEC.yaml, math definition.
**Outputs:** golden/*_golden.hpp, golden/*_golden.cpp.
**Failure modes:** Numerically unstable reductions, incorrect broadcasting, wrong dtype promotion, missing edge cases, reliance on external frameworks.
**Verification checklist:**
- [ ] Pure C++ implementation, no framework dependency
- [ ] Deterministic for given inputs
- [ ] Float32 accumulation for reductions where appropriate
- [ ] Numerically stable softmax/reduction (max-subtract, etc.)
- [ ] Handles all shape/dtype combinations from spec
- [ ] Edge cases: zeros, ones, large values, small values

## Role 6: Test Engineer

**Responsibility:** Write and maintain GTest/CTest coverage.
**Inputs:** SPEC.yaml, CPU golden, operator implementation.
**Outputs:** tests/test_<op>.cpp, CTest labels, test data generators.
**Failure modes:** Tests pass without checking values, missing edge shapes, missing dtype coverage, non-deterministic tests, silent skips.
**Verification checklist:**
- [ ] Shape coverage from spec
- [ ] Dtype coverage from spec
- [ ] Edge case tests (empty, singleton, large, non-power-of-2)
- [ ] Invalid input rejection tests
- [ ] Golden comparison with correct tolerances
- [ ] GTest skip for NPU unavailable
- [ ] CTest labels: host, golden, npu, <category>
- [ ] Deterministic seed printed on failure

## Role 7: Performance Engineer

**Responsibility:** Benchmark and tune operator performance.
**Inputs:** SPEC.yaml performance targets, operator implementation, NPU hardware.
**Outputs:** PERF.md, benchmark.json, tuning recommendations.
**Failure modes:** No warmup, insufficient iterations, ignoring target comparison, claiming pass without measurement, missing regression detection.
**Verification checklist:**
- [ ] Warmup iterations before measurement
- [ ] Sufficient iterations for statistical validity
- [ ] Average/min/max reported
- [ ] Bandwidth or FLOPs where meaningful
- [ ] Target comparison explicit (pass/fail/gap)
- [ ] Before/after comparison for tuning iterations
- [ ] Regression report if performance degraded

## Role 8: Systematic Debugger

**Responsibility:** Debug build, test, accuracy, or benchmark failures.
**Inputs:** Failure evidence, build logs, test output, accuracy reports.
**Outputs:** Root cause analysis, fix, rerun evidence.
**Failure modes:** Fixing symptoms not causes, introducing regressions, not recording evidence, skipping reproduction.
**Verification checklist:**
- [ ] Failure reproduced
- [ ] Failure isolated to specific component
- [ ] Root cause explained in terms of code/spec mismatch
- [ ] One cause fixed at a time
- [ ] Fix verified by rerunning the failing test
- [ ] No regression in other tests

## Role 9: Reviewer

**Responsibility:** Review operator implementation before shipping.
**Inputs:** SPEC.yaml, implementation, tests, accuracy report, performance report.
**Outputs:** Review checklist, blocking/non-blocking issues.
**Failure modes:** Rubber-stamping, missing spec conformance checks, ignoring performance gaps, approving with failing tests.
**Verification checklist:**
- [ ] Spec conformance: all spec requirements met
- [ ] Code quality: no obvious bugs, clear structure
- [ ] Test coverage: shapes, dtypes, edge cases covered
- [ ] Accuracy: tolerances met for all tested configs
- [ ] Performance: targets met or gaps documented
- [ ] Documentation: operator doc complete and accurate
- [ ] Registry: OPERATOR_REGISTRY.yaml updated

## Role 10: Shipping Reporter

**Responsibility:** Produce final shipping report for an operator.
**Inputs:** All operator artifacts (spec, code, tests, results, perf).
**Outputs:** Final report with status, evidence, blockers, next steps.
**Failure modes:** Claiming success without evidence, hiding blockers, incomplete artifact inventory, missing commands to reproduce.
**Verification checklist:**
- [ ] Spec exists and is valid
- [ ] Docs exist and are accurate
- [ ] CPU golden exists and passes
- [ ] Tests pass (host, NPU if available)
- [ ] Accuracy report exists
- [ ] Benchmark report exists if required
- [ ] Blocker report exists if anything incomplete
- [ ] All commands to reproduce are documented
