# /npu-doctor

Diagnose a broken or misconfigured Harness environment. Check toolchain availability, build health, test status, spec validity, and state file consistency.

## Usage

```
/npu-doctor
```

No arguments. Runs a comprehensive diagnostic sweep.

## When to Use

- Build fails with unclear errors
- Tests fail unexpectedly
- CANN was upgraded or changed paths
- After system updates
- When `/npu-status` shows unexpected results
- When switching between machines
- When the environment "used to work" but doesn't now

## Required Commands

Run these commands first. Their output is the primary diagnostic data.

```bash
python3 scripts/detect_ascend_env.py
```

This detects the CANN environment. If it fails, note the error — host-only mode should still work.

```bash
python3 scripts/harness.py doctor
```

This runs the harness self-diagnostic. If the script does not exist or does not support `doctor`, perform manual diagnostics (see below).

## Diagnostic Checklist

Run through every item. Report PASS, WARN, or FAIL for each.

### 1. Toolchain Health

```bash
# Check cmake
cmake --version 2>&1 || echo "FAIL: cmake not found"

# Check ninja
ninja --version 2>&1 || echo "FAIL: ninja not found"

# Check g++
g++ --version 2>&1 || echo "FAIL: g++ not found"

# Check python3
python3 --version 2>&1 || echo "FAIL: python3 not found"
```

| Tool | Required | Status |
|------|----------|--------|
| cmake | Yes (host + NPU builds) | <version / FAIL> |
| ninja | Yes (build system) | <version / FAIL> |
| g++ | Yes (C++ compilation) | <version / FAIL> |
| python3 | Yes (scripts/CLI) | <version / FAIL> |
| GTest | Yes (testing) | <found / FAIL> |
| ccec | Only for NPU | <version / not found> |
| npu-smi | Only for NPU device query | <found / not found> |

### 2. CANN Environment

From `detect_ascend_env.py` output, verify:

```bash
# Check CANN installation
ls $ASCEND_HOME 2>/dev/null || echo "WARN: ASCEND_HOME not set or directory not found"

# Check compiler
which ccec 2>/dev/null || echo "WARN: ccec not in PATH"

# Check NPU devices (only if CANN detected)
npu-smi info 2>/dev/null || echo "WARN: npu-smi not available or no devices"
```

| Component | Status | Action if Missing |
|-----------|--------|-------------------|
| ASCEND_HOME | <set/not set> | Set environment variable or install CANN |
| CANN version | <version/not detected> | Install CANN toolkit |
| ccec compiler | <path/not found> | Check CANN installation; host-only mode still works |
| Ascend headers | <path/not found> | Check CANN include paths |
| Ascend libraries | <path/not found> | Check CANN library paths |
| NPU driver | <found/not found> | Install driver or check permissions |
| NPU devices | <count/none> | Check device availability; tests will skip at runtime |

### 3. Build Health

```bash
# Check if build directories exist
ls build/host/CMakeCache.txt 2>/dev/null || echo "WARN: build/host not configured"
ls build/npu/CMakeCache.txt 2>/dev/null || echo "INFO: build/npu not configured (OK if NPU not needed)"

# Check CMake cache for key variables (host)
grep -E "HARNESS_ENABLE_ASCEND|CMAKE_BUILD_TYPE|HARNESS_NPU_ARCH" build/host/CMakeCache.txt 2>/dev/null || echo "INFO: No host CMake cache"
```

| Check | Status | Action if FAIL |
|-------|--------|----------------|
| build/host configured | <yes/no> | Run: `cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo` |
| build/host built | <yes/no> | Run: `cmake --build build/host` |
| build/npu configured | <yes/no> | Run NPU cmake configure (only if CANN available) |
| build/npu built | <yes/no> | Run: `cmake --build build/npu` |
| HARNESS_ENABLE_ASCEND=OFF (host) | <yes/no> | Reconfigure host build correctly |
| HARNESS_NPU_ARCH=dav-2201 (npu) | <yes/no> | Reconfigure NPU build with correct arch |

### 4. Build Artifacts

Check for stale or missing build artifacts:

```bash
# Check for stale CMake cache (different generator or settings)
ls build/*/CMakeCache.txt 2>/dev/null

# Check for missing Ninja build files
ls build/*/*.ninja 2>/dev/null || echo "WARN: No .ninja files found in build directories"
```

| Check | Action if Issue Found |
|-------|----------------------|
| CMake cache from different generator | Delete build directory and reconfigure |
| Missing compiled objects | Rebuild: `cmake --build build/host` |
| Stale objects (source changed, not rebuilt) | Rebuild: `cmake --build build/host` |
| Multiple conflicting build directories | Identify correct one and remove others |

### 5. Test Health

```bash
# Run host tests and capture failures
ctest --test-dir build/host --output-on-failure 2>&1 || echo "WARN: Host tests have failures"

# Check for missing test registration
ctest --test-dir build/host -N 2>&1 | head -5 || echo "WARN: No tests registered in host build"

# If NPU build exists
ctest --test-dir build/npu --output-on-failure 2>&1 || echo "INFO: NPU tests have failures or NPU build missing"
```

| Check | Action if FAIL |
|-------|----------------|
| No tests registered | Check CMakeLists.txt test registration; add CTest labels |
| Tests fail to build | Fix compilation errors in test files |
| Tests segfault | Debug with gdb; check for null pointers, buffer overflows |
| Tests timeout | Increase CTest timeout or fix infinite loop |
| NPU tests fail | Check device availability; compare kernel to golden |
| Flaky tests | Add deterministic seed; fix race conditions |

### 6. Spec File Validity

Check all spec files for structural validity:

```bash
# List all spec files
ls specs/*.yaml 2>/dev/null || echo "INFO: No spec files in specs/"

# Check each spec for required fields
for f in specs/*.yaml; do
  echo "=== $f ==="
  python3 -c "
import yaml
with open('$f') as fh:
    spec = yaml.safe_load(fh)
required = ['name', 'category', 'inputs', 'outputs', 'accuracy']
for r in required:
    if r not in spec:
        print(f'MISSING: {r}')
    elif not spec[r]:
        print(f'EMPTY: {r}')
" 2>&1 || echo "WARN: Could not validate $f"
done
```

| Check | Action if Issue Found |
|-------|----------------------|
| Missing spec file | Run `/npu-new-op` for the operator |
| Invalid YAML syntax | Fix YAML syntax errors |
| Missing required fields (name, category, inputs, outputs, accuracy) | Run `/npu-discuss-op` to fill in missing fields |
| Empty required fields (e.g., empty inputs list) | Run `/npu-discuss-op` to fill in missing details |
| Contradictory shape rules | Fix logic errors in shape_rules |
| Unrealistic tolerances | Adjust accuracy tolerances to match dtypes |
| Performance targets without metric | Add metric (latency_us, bandwidth_gb_s, etc.) |

### 7. State File Consistency

Check for inconsistencies between state files:

```bash
# Check OPERATOR_REGISTRY.yaml exists
ls OPERATOR_REGISTRY.yaml 2>/dev/null || echo "FAIL: OPERATOR_REGISTRY.yaml not found"

# Check registry vs actual files
for op_dir in .npu-harness/operators/*/; do
    op_name=$(basename "$op_dir")
    echo "Checking $op_name..."
    # Check spec exists
    [ -f "$op_dir/SPEC.yaml" ] || echo "  WARN: Missing SPEC.yaml"
    # Check status exists
    [ -f "$op_dir/STATUS.md" ] || echo "  WARN: Missing STATUS.md"
done
```

| Check | Action if Issue Found |
|-------|----------------------|
| OPERATOR_REGISTRY.yaml missing | Run `/npu-map-codebase` |
| Registry has operators not in .npu-harness/ | Run `/npu-map-codebase` to discover them |
| .npu-harness/operators/<op>/ has no registry entry | Update registry manually or run `/npu-map-codebase` |
| STATUS.md phase claims "done" but files are missing | Re-run the relevant implementation step |
| Registry says "tested" but no RESULTS.md | Run `/npu-verify-op` |
| Multiple spec files for same operator | Choose the canonical one; remove duplicate |
| .npu-harness/state/ copies out of date | Copy fresh from root: `cp STATE.md .npu-harness/state/STATE.md` |

### 8. Generated File Freshness

Check if generated files are stale (older than their source spec):

```bash
# Compare modification times
for op_dir in .npu-harness/operators/*/; do
    op_name=$(basename "$op_dir")
    spec_file="$op_dir/SPEC.yaml"
    gen_file="ops/$op_name/CMakeLists.txt"
    if [ -f "$spec_file" ] && [ -f "$gen_file" ]; then
        if [ "$spec_file" -nt "$gen_file" ]; then
            echo "WARN: $op_name — SPEC.yaml is newer than generated CMakeLists.txt (regeneration needed)"
        fi
    fi
done
```

| Check | Action if Issue Found |
|-------|----------------------|
| SPEC.yaml newer than generated files | Re-run spec generator: `python3 scripts/generate_op_from_spec.py specs/<op_name>.yaml` |
| Generated files exist but spec was deleted | Remove stale generated files or recreate spec |
| Manual edits in generated files (no generated marker) | Preserve manual edits; do not regenerate without `--force` |

### 9. Filesystem Permissions

```bash
# Check write permission on key directories
touch .npu-harness/.write_test 2>/dev/null && rm .npu-harness/.write_test || echo "FAIL: Cannot write to .npu-harness/"
touch build/.write_test 2>/dev/null && rm build/.write_test || echo "WARN: Cannot write to build/ (may need sudo or directory creation)"
```

| Check | Action if Issue Found |
|-------|----------------------|
| Cannot write to .npu-harness/ | Fix permissions: `chmod -R u+w .npu-harness/` |
| Cannot write to build/ | Fix permissions or use different build directory |
| Cannot execute scripts/ | `chmod +x scripts/*.py scripts/*.sh` |

## Summary Report

After running all checks, produce a summary:

```
============================
 NPU Harness Doctor Report
============================

Date: <today>

Checks:
  Toolchain:   <N> pass, <M> warn, <K> fail
  CANN:        <N> pass, <M> warn, <K> fail
  Build:       <N> pass, <M> warn, <K> fail
  Tests:       <N> pass, <M> warn, <K> fail
  Specs:       <N> pass, <M> warn, <K> fail
  State files: <N> pass, <M> warn, <K> fail
  Freshness:   <N> pass, <M> warn, <K> fail
  Permissions: <N> pass, <M> warn, <K> fail

Overall: <HEALTHY|DEGRADED|BROKEN>

Critical Issues (must fix):
  1. <issue> — <fix command>
  2. <issue> — <fix command>

Warnings (should fix):
  1. <issue> — <fix command>
  2. <issue> — <fix command>

Recommendations:
  1. <concrete action>
  2. <concrete action>
```

### Health Determination

- **HEALTHY:** All checks pass or have only informational messages. Build and test work.
- **DEGRADED:** Some warnings but host build and tests still work. NPU path may be unavailable.
- **BROKEN:** Failures in toolchain, build, or tests prevent core Harness operation.

## Rules

1. Check everything. Do not skip checks just because "it's probably fine."
2. Every FAIL must have a concrete fix command or action. Do not just report problems.
3. If a tool is missing, suggest how to install it. Be specific for the platform (Linux, assumed).
4. Do not modify files during diagnosis without explicit user permission. The doctor is read-only by default.
5. If the environment is HEALTHY, say so clearly and concisely. Do not pad the report.
6. If the environment is BROKEN, prioritize fixes: toolchain first, then build, then tests, then state files.
7. Host-only mode should ALWAYS be achievable. If the host build is broken, that is a critical issue.
8. Missing CANN is a warning for NPU mode but normal for host-only. Do not report it as a failure.

## Output

The doctor command outputs the full diagnostic report. At minimum, the user should know:
- Whether the Harness is healthy, degraded, or broken
- What the critical issues are (and how to fix them)
- What the warnings are (and how to fix them)
- What command to run next

If everything is healthy:

```
Harness is HEALTHY. Host build and tests pass. NPU available.

Next: /npu-status to see operator status table.
```
