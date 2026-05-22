# /npu-new-project

Initialize the Harness state and baseline project files. This command sets up the agentic Harness layer from scratch or reinitializes it.

## When to Use

- First time setting up the Harness in a repository.
- After a major refactor that requires state re-initialization.
- When the `.npu-harness/` directory is missing or corrupted.

Do NOT use this command on an already-initialized Harness unless you intend to overwrite state files. It will create fresh copies of all state files.

## Prerequisites

Before running, verify:

- You are in the repository root (`/home/developer/workspace/harness`)
- CMake and Ninja are installed
- GTest is available
- Python 3 is available
- `scripts/harness.py` exists (or will be created)

## Files to Create

### Top-Level State Files

Create or overwrite these files with Harness-appropriate content. If they already exist and have useful content, update them rather than replacing wholesale.

1. **PROJECT.md** — Mission statement, two-layer architecture description, target hardware, project style, 10-operator bootstrap ladder, quality bar.

2. **REQUIREMENTS.md** — Functional requirements (FR1-FR10), non-functional requirements (NFR1-NFR4), per-operator requirements. Reference: FR1 = operator intake, FR2 = spec-driven dev, FR3 = CPU golden, FR4 = GTest/CTest, FR5 = host-only mode, FR6 = NPU mode, FR7 = direct launch, FR8 = performance contract, FR9 = state persistence, FR10 = operator registry.

3. **ROADMAP.md** — Five-phase roadmap: Foundation, Agentic Layer, Operator Bootstrap 1-5, Operator Bootstrap 6-10, Verification.

4. **STATE.md** — Current Harness state snapshot with environment table, Harness layer status table, build status table, and a pointer to OPERATOR_REGISTRY.yaml.

5. **CONTEXT.md** — Operational context: what the Harness is, what it is not, key constraints, working instructions, CANN environment info, repository layout diagram.

### Harness Config

Create `.npu-harness/config.yaml`:

```yaml
harness_version: 1
target_arch: dav-2201
build_system: cmake+ninja
test_framework: gtest+ctest
launch_mode: direct
modes:
  host:
    enabled: true
    cmake_args:
      - -DHARNESS_ENABLE_ASCEND=OFF
      - -DCMAKE_BUILD_TYPE=RelWithDebInfo
    build_dir: build/host
  npu:
    enabled: auto
    cmake_args:
      - -DHARNESS_ENABLE_ASCEND=ON
      - -DHARNESS_NPU_ARCH=dav-2201
      - -DCMAKE_BUILD_TYPE=RelWithDebInfo
    build_dir: build/npu
default_tolerances:
  float32:
    abs: 1.0e-5
    rel: 1.0e-5
  float16:
    abs: 1.0e-2
    rel: 1.0e-2
benchmark_defaults:
  warmup: 10
  iterations: 100
```

### State Directory Copies

Copy the top-level state files into `.npu-harness/state/`:

```bash
cp PROJECT.md .npu-harness/state/PROJECT.md
cp REQUIREMENTS.md .npu-harness/state/REQUIREMENTS.md
cp ROADMAP.md .npu-harness/state/ROADMAP.md
cp STATE.md .npu-harness/state/STATE.md
cp CONTEXT.md .npu-harness/state/CONTEXT.md
```

### Templates

Create these template files under `.npu-harness/templates/`:

- `operator_spec.yaml` — Skeleton YAML spec for new operators
- `operator_plan.md` — Skeleton implementation plan
- `operator_status.md` — Skeleton status tracker
- `final_report.md` — Skeleton shipping report

Templates should contain placeholder fields marked with `{{ PLACEHOLDER }}` syntax.

## Host-Only Baseline Verification

After creating all files, verify that the host-only build works:

```bash
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

If configure fails, diagnose and fix before proceeding.

```bash
cmake --build build/host
```

If build fails, diagnose and fix before proceeding.

```bash
ctest --test-dir build/host --output-on-failure
```

If no tests are registered yet, this is acceptable. The command should exit cleanly (no tests found is not a failure).

## Verification Checklist

Before declaring `/npu-new-project` complete:

- [ ] `PROJECT.md` exists and has mission, architecture, operator ladder
- [ ] `REQUIREMENTS.md` exists with FR1-FR10 and NFR1-NFR4
- [ ] `ROADMAP.md` exists with five phases
- [ ] `STATE.md` exists with environment table
- [ ] `CONTEXT.md` exists with operational context
- [ ] `.npu-harness/config.yaml` exists with valid YAML
- [ ] `.npu-harness/state/` contains copies of all five state files
- [ ] `.npu-harness/templates/` contains operator spec, plan, status, and report templates
- [ ] `.npu-harness/operators/` directory exists (may be empty)
- [ ] `.npu-harness/runs/` directory exists (may be empty)
- [ ] `cmake -S . -B build/host` configures successfully
- [ ] `cmake --build build/host` builds successfully
- [ ] `ctest --test-dir build/host` runs without error (zero tests is OK)

## Output

At the end, print a summary listing every file created and the result of the host baseline build.
