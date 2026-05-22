# CLAUDE.md — NPU Operator Development Harness

You are working inside an NPU operator development Harness for Ascend NPU (dav-2201).

## Core behavior

Before implementing any operator:
1. Read PROJECT.md, REQUIREMENTS.md, ROADMAP.md, STATE.md, CONTEXT.md.
2. Read OPERATOR_REGISTRY.yaml.
3. If working on an operator, read .npu-harness/operators/<op>/SPEC.yaml and PLAN.md.
4. Use the relevant skills under .claude/skills.
5. Do not start with code until the operator contract is clear.
6. Always produce CPU golden first or alongside the kernel.
7. Always add GTest coverage.
8. Always run host tests.
9. If NPU is available, run NPU tests.
10. If performance target exists, run benchmarks and compare to target.
11. Do not declare success without evidence.
12. Do not treat missing NPU as host-only failure.
13. Do not hide blockers.
14. Default NPU arch is dav-2201.

## Default operator workflow

```
intake -> spec -> discuss -> plan -> execute -> verify -> tune -> ship
```

## Slash commands

- `/npu-map-codebase` — Analyze repo, detect environment, update state
- `/npu-new-project` — Initialize Harness state and baseline files
- `/npu-new-op <<<...>>>` — Accept natural-language operator requirement
- `/npu-discuss-op <op_name>` — Resolve ambiguity before planning
- `/npu-plan-op <op_name>` — Create implementation plan
- `/npu-execute-op <op_name>` — Implement operator (RED → GREEN → REFACTOR)
- `/npu-verify-op <op_name>` — Prove correctness
- `/npu-tune-op <op_name>` — Benchmark and tune performance
- `/npu-ship-op <op_name>` — Finalize, update registry, produce shipping report
- `/npu-status` — Print Harness state and operator status table
- `/npu-doctor` — Diagnose broken environment

## Build commands

Host-only:
```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

NPU mode:
```bash
cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/npu
ctest --test-dir build/npu --output-on-failure
```

## Harness CLI

```bash
python3 scripts/harness.py init
python3 scripts/harness.py map-codebase
python3 scripts/harness.py status
python3 scripts/harness.py doctor
python3 scripts/harness.py new-op --spec specs/<op>.yaml
python3 scripts/harness.py plan <op_name>
python3 scripts/harness.py generate <op_name>
python3 scripts/harness.py build --mode host
python3 scripts/harness.py build --mode npu
python3 scripts/harness.py test --mode host --op <op_name>
python3 scripts/harness.py test --mode npu --op <op_name>
python3 scripts/harness.py bench <op_name>
python3 scripts/harness.py verify <op_name>
python3 scripts/harness.py verify-performance <op_name>
python3 scripts/harness.py ship <op_name>
```
