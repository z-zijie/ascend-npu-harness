# /npu-map-codebase

Analyze the repository, detect the build system, detect the CANN environment, discover existing operators, and update the Harness state files.

## When to Use

- After first cloning the repository.
- After switching branches or pulling upstream changes.
- After installing or upgrading CANN.
- After adding new operators outside the Harness workflow.
- Whenever STATE.md or OPERATOR_REGISTRY.yaml appears stale.

## What to Read First

Before running commands, read these files to understand the current baseline:

1. `/home/developer/workspace/harness/PROJECT.md`
2. `/home/developer/workspace/harness/REQUIREMENTS.md`
3. `/home/developer/workspace/harness/ROADMAP.md`
4. `/home/developer/workspace/harness/STATE.md`
5. `/home/developer/workspace/harness/CONTEXT.md`

## Required Commands

Run these commands exactly as written. Capture their output.

```bash
pwd
```

```bash
find . -maxdepth 4 -type f \
  -not -path '*/.git/*' \
  -not -path '*/build/*' \
  -not -path '*/.claude/*' \
  | sort
```

```bash
python3 scripts/detect_ascend_env.py || true
```

```bash
python3 scripts/harness.py status || true
```

## Inventory Discovery

After collecting command output, perform these discovery steps:

### 1. Build System Detection

Check for the presence and health of:

- `CMakeLists.txt` (root CMake)
- `cmake/` modules (Ascend.cmake, CompilerOptions.cmake, GTest.cmake, OperatorTarget.cmake)
- CMake presets (`CMakePresets.json`)
- Ninja availability (`ninja --version`)

### 2. CANN Environment Detection

Use the output of `detect_ascend_env.py` and verify:

- CANN version and install path
- `ccec` compiler availability
- Include paths and library paths
- NPU device visibility (`npu-smi info`)
- Runtime library availability

### 3. Operator Discovery

Scan for existing operators by checking:

- `specs/` directory for YAML spec files
- `ops/` directory for operator implementation directories
- `tests/` directory for `test_<op>.cpp` files
- `.npu-harness/operators/` for per-operator state
- Any existing `OPERATOR_REGISTRY.yaml`

### 4. Source File Inventory

Categorize discovered files into:

- Core Harness headers (`include/harness/`)
- Core Harness sources (`src/`)
- Operator implementations (`ops/<op_name>/`)
- Test files (`tests/`)
- Scripts (`scripts/`)
- Docs (`docs/`)

## Files to Update

### STATE.md

Update the environment table with fresh detection results:

```markdown
## Environment

| Component | Status | Details |
|-----------|--------|---------|
| cmake | <version> | <path or 'not found'> |
| ninja | <version> | <path or 'not found'> |
| g++ | <version> | <path or 'not found'> |
| python3 | <version> | <path or 'not found'> |
| GTest | <version> | <path or 'not found'> |
| CANN | <version> | <path or 'not detected'> |
| NPU devices | <count> | <device info or 'none'> |
| Target arch | dav-2201 | Default |
```

Set `Last updated:` to today's date.

### CONTEXT.md

Update the CANN Environment section with the current detection results. Preserve all other sections. If new build modes or scripts are discovered, add them to the Repository Layout section or the build instructions.

### OPERATOR_REGISTRY.yaml

Create or update with entries for every discovered operator. For each operator, include at minimum:

```yaml
schema_version: 1
target_arch_default: dav-2201
operators:
  <op_name>:
    category: <elementwise|reduction|layout|matmul|normalization|attention|fusion|custom>
    spec: specs/<op_name>.yaml
    status_file: .npu-harness/operators/<op_name>/STATUS.md
    generated: <true|false>
    built_host: <true|false|unknown>
    tested_host: <true|false|unknown>
    built_npu: <true|false|skipped>
    tested_npu: <true|false|skipped>
    npu_skip_reason: "<reason if skipped>"
    accuracy: <pass|fail|pending|unknown>
    performance: <pass|fail|sanity|pending|pending_npu|not_required>
```

### Run Report

Create a timestamped run directory and write a summary report:

```bash
mkdir -p .npu-harness/runs/$(date +%Y%m%d_%H%M%S)
```

Write `.npu-harness/runs/<timestamp>/report.md` containing:

- Command outputs (stdout/stderr)
- Discovery summary
- Files created or updated
- Any warnings or issues found
- Next recommended commands

## Verification Checklist

Before declaring `/npu-map-codebase` complete:

- [ ] `pwd` confirms correct working directory
- [ ] File inventory captured (find output)
- [ ] `detect_ascend_env.py` ran (or true'd gracefully)
- [ ] `harness.py status` ran (or true'd gracefully)
- [ ] `STATE.md` has fresh date and accurate environment table
- [ ] `CONTEXT.md` has current CANN environment section
- [ ] `OPERATOR_REGISTRY.yaml` exists and covers all discovered operators
- [ ] Timestamped run report written to `.npu-harness/runs/<timestamp>/report.md`
- [ ] No file was deleted or truncated

## Failure Handling

- If `detect_ascend_env.py` does not exist, note it in the report and skip. Host-only mode is still valid.
- If `harness.py` does not exist, note it as a missing script and continue with manual discovery.
- If no operators are found, record an empty operator list in the registry. This is not an error.
- If CANN is not detected, mark NPU fields as `skipped` with reason `"CANN not detected"`.

## Output

At the end, print a summary of what was discovered and what files were updated.
