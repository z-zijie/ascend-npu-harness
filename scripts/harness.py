#!/usr/bin/env python3
"""
NPU Operator Development Harness CLI.

Usage:
  python3 scripts/harness.py init
  python3 scripts/harness.py map-codebase
  python3 scripts/harness.py status
  python3 scripts/harness.py doctor
  python3 scripts/harness.py new-op --spec specs/<op>.yaml
  python3 scripts/harness.py plan <op_name>
  python3 scripts/harness.py generate <op_name>
  python3 scripts/harness.py build --mode host|npu
  python3 scripts/harness.py test --mode host|npu [--op <op_name>]
  python3 scripts/harness.py bench <op_name>
  python3 scripts/harness.py verify <op_name>
  python3 scripts/harness.py verify-performance <op_name>
  python3 scripts/harness.py ship <op_name>
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = PROJECT_ROOT / ".npu-harness"
STATE_DIR = HARNESS_DIR / "state"
OPS_DIR = HARNESS_DIR / "operators"
RUNS_DIR = HARNESS_DIR / "runs"
SPECS_DIR = PROJECT_ROOT / "specs"
BUILD_HOST_DIR = PROJECT_ROOT / "build" / "host"
BUILD_NPU_DIR = PROJECT_ROOT / "build" / "npu"
REGISTRY_FILE = PROJECT_ROOT / "OPERATOR_REGISTRY.yaml"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd, cwd=None, capture=True, timeout=300):
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"  RUN: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True,
            cwd=cwd or PROJECT_ROOT, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -1, "", str(e)


def make_run_dir(name):
    d = RUNS_DIR / f"{timestamp()}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def cmd_init(args):
    """Initialize Harness state files."""
    print("=== NPU Harness: init ===")
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "harness_version": "0.1.0",
        "default_arch": "dav-2201",
        "created": timestamp(),
        "modes": ["host", "npu"],
    }
    save_yaml(HARNESS_DIR / "config.yaml", config)

    # Initialize registry if not exists
    if not REGISTRY_FILE.exists():
        registry = {
            "schema_version": 1,
            "target_arch_default": "dav-2201",
            "operators": {}
        }
        save_yaml(REGISTRY_FILE, registry)

    print("Harness initialized.")
    return 0


def cmd_map_codebase(args):
    """Analyze and update state."""
    print("=== NPU Harness: map-codebase ===")
    run_dir = make_run_dir("map-codebase")

    # Run environment detection
    ret, out, err = run_cmd("python3 scripts/detect_ascend_env.py")
    log_file = run_dir / "detect_env.log"
    log_file.write_text(out + "\n" + err)

    # File listing
    ret2, out2, _ = run_cmd("find . -maxdepth 4 -type f | sort")
    (run_dir / "file_listing.txt").write_text(out2)

    # Update state
    state = {
        "last_mapped": timestamp(),
        "environment_detected": True,
    }
    save_json(run_dir / "report.json", state)

    print(f"Map-codebase complete. Run dir: {run_dir}")
    return 0


def cmd_status(args):
    """Print operator status table."""
    print("=== NPU Harness: status ===")
    if not REGISTRY_FILE.exists():
        print("No operator registry found. Run 'init' first.")
        return 1

    registry = load_yaml(REGISTRY_FILE)
    ops = registry.get("operators", {})

    print(f"\n{'='*100}")
    print(f"{'#':<3} {'Operator':<30} {'Generated':<10} {'Built':<8} {'HostTest':<10} {'NPUTest':<10} {'Accuracy':<10} {'Perf':<12} {'Notes'}")
    print(f"{'='*100}")

    for i, (name, info) in enumerate(ops.items(), 1):
        gen = "yes" if info.get("generated") else "no"
        built = "yes" if info.get("built_host") else "no"
        host_t = "pass" if info.get("tested_host") else "pending"
        npu_t = info.get("tested_npu", "pending")
        acc = info.get("accuracy", "pending")
        perf = info.get("performance", "pending")
        notes = info.get("notes", "")
        print(f"{i:<3} {name:<30} {gen:<10} {built:<8} {host_t:<10} {npu_t:<10} {acc:<10} {perf:<12} {notes}")

    print(f"{'='*100}")
    return 0


def cmd_doctor(args):
    """Diagnose environment."""
    print("=== NPU Harness: doctor ===")
    issues = []

    # Check cmake
    ret, out, _ = run_cmd("cmake --version 2>&1 | head -1")
    if ret != 0:
        issues.append("cmake not found")
    else:
        print(f"  cmake: OK ({out.strip()})")

    # Check ninja
    ret, out, _ = run_cmd("ninja --version 2>&1")
    if ret != 0:
        issues.append("ninja not found")
    else:
        print(f"  ninja: OK ({out.strip()})")

    # Check g++
    ret, out, _ = run_cmd("g++ --version 2>&1 | head -1")
    if ret != 0:
        issues.append("g++ not found")
    else:
        print(f"  g++: OK ({out.strip()})")

    # Check GTest headers
    gtest_h = PROJECT_ROOT / ".." / ".." / "usr" / "include" / "gtest" / "gtest.h"
    if not os.path.exists("/usr/include/gtest/gtest.h"):
        issues.append("GTest headers not found at /usr/include/gtest/")
    else:
        print("  GTest headers: OK")

    # Check CANN
    cann_home = os.environ.get("ASCEND_HOME_PATH")
    if cann_home and os.path.exists(cann_home):
        print(f"  CANN: OK ({cann_home})")
    else:
        print("  CANN: not detected (host-only mode available)")

    # Check NPU devices
    import glob
    devices = glob.glob("/dev/davinci*")
    if devices:
        print(f"  NPU devices: OK ({len(devices)} found)")
    else:
        print("  NPU devices: none found (NPU tests will skip)")

    # Check registry
    if REGISTRY_FILE.exists():
        print("  Operator registry: OK")
    else:
        issues.append("OPERATOR_REGISTRY.yaml missing")

    if issues:
        print(f"\n  Issues found ({len(issues)}):")
        for i in issues:
            print(f"    - {i}")
    else:
        print("\n  No issues found.")

    return len(issues)


def cmd_build(args):
    """Build the project."""
    mode = args.mode or "host"
    print(f"=== NPU Harness: build --mode {mode} ===")
    run_dir = make_run_dir(f"build_{mode}")

    if mode == "host":
        build_dir = BUILD_HOST_DIR
        cmake_cmd = (
            f"cmake -S . -B {build_dir} -G Ninja "
            f"-DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo"
        )
    else:
        build_dir = BUILD_NPU_DIR
        cmake_cmd = (
            f"cmake -S . -B {build_dir} -G Ninja "
            f"-DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 "
            f"-DCMAKE_BUILD_TYPE=RelWithDebInfo"
        )

    # Configure
    ret, out, err = run_cmd(cmake_cmd)
    (run_dir / "configure.log").write_text(out + "\n" + err)
    if ret != 0:
        print(f"Configure FAILED:\n{err}")
        (run_dir / "build.log").write_text("Configure failed")
        return ret

    # Build
    ret, out, err = run_cmd(f"cmake --build {build_dir}")
    (run_dir / "build.log").write_text(out + "\n" + err)
    if ret != 0:
        print(f"Build FAILED:\n{err}")
        return ret

    print(f"Build SUCCESS. Run dir: {run_dir}")
    return 0


def cmd_test(args):
    """Run tests."""
    mode = args.mode or "host"
    op_filter = args.op
    print(f"=== NPU Harness: test --mode {mode} --op {op_filter or 'all'} ===")
    run_dir = make_run_dir(f"test_{mode}")

    build_dir = BUILD_HOST_DIR if mode == "host" else BUILD_NPU_DIR

    ctest_cmd = f"ctest --test-dir {build_dir} --output-on-failure"
    if op_filter:
        ctest_cmd += f" -R {op_filter}"

    ret, out, err = run_cmd(ctest_cmd)
    (run_dir / "ctest.log").write_text(out + "\n" + err)

    print(out[-2000:] if len(out) > 2000 else out)
    if ret != 0:
        print(f"\nTests FAILED (exit code {ret})")
    else:
        print(f"\nTests PASSED")

    return ret


def cmd_generate(args):
    """Generate operator from spec."""
    op_name = args.op_name
    print(f"=== NPU Harness: generate {op_name} ===")

    spec_file = SPECS_DIR / f"{op_name}.yaml"
    # Try numbered prefix
    if not spec_file.exists():
        for f in SPECS_DIR.glob("*.yaml"):
            data = load_yaml(f)
            if data and data.get("name") == op_name:
                spec_file = f
                break

    if not spec_file.exists():
        print(f"Spec not found for operator: {op_name}")
        return 1

    # Run generator
    ret, out, err = run_cmd(f"python3 scripts/generate_op_from_spec.py {spec_file}")
    print(out)
    if ret != 0:
        print(f"Generate FAILED:\n{err}")
        return ret
    return 0


def cmd_bench(args):
    """Run benchmark for operator."""
    op_name = args.op_name
    print(f"=== NPU Harness: bench {op_name} ===")
    run_dir = make_run_dir(f"bench_{op_name}")

    # Try to run host benchmark
    bench_exe = BUILD_HOST_DIR / "ops" / op_name / "bench" / f"bench_{op_name}"
    if bench_exe.exists():
        ret, out, err = run_cmd(str(bench_exe))
        (run_dir / "benchmark.log").write_text(out + "\n" + err)
        print(out[-2000:] if len(out) > 2000 else out)
    else:
        print(f"No benchmark executable found for {op_name}")
        # Create stub benchmark report
        bench_data = {
            "operator": op_name,
            "status": "no_benchmark_target",
            "note": "No benchmark executable found; host-only build may not include benchmarks"
        }
        save_json(run_dir / "benchmark.json", bench_data)

    return 0


def cmd_verify(args):
    """Verify operator correctness."""
    op_name = args.op_name
    print(f"=== NPU Harness: verify {op_name} ===")
    run_dir = make_run_dir(f"verify_{op_name}")

    # Run host tests for this operator
    ret, out, err = run_cmd(
        f"ctest --test-dir {BUILD_HOST_DIR} -R {op_name} --output-on-failure"
    )
    (run_dir / "accuracy.json").write_text(
        json.dumps({"operator": op_name, "host_tests_passed": ret == 0,
                     "exit_code": ret}, indent=2)
    )

    print(out[-2000:] if len(out) > 2000 else out)

    # Update operator status
    op_dir = OPS_DIR / op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "operator": op_name,
        "verified_at": timestamp(),
        "host_tests_passed": ret == 0,
    }
    save_json(op_dir / "RESULTS.md", status)
    # Also save as JSON for machine reading
    save_json(op_dir / "RESULTS.json", status)

    return ret


def cmd_verify_performance(args):
    """Verify performance against targets."""
    op_name = args.op_name
    print(f"=== NPU Harness: verify-performance {op_name} ===")

    spec_file = SPECS_DIR / f"{op_name}.yaml"
    if not spec_file.exists():
        for f in SPECS_DIR.glob("*.yaml"):
            data = load_yaml(f)
            if data and data.get("name") == op_name:
                spec_file = f
                break

    if not spec_file.exists():
        print(f"No spec found for {op_name}")
        return 1

    spec = load_yaml(spec_file)
    perf = spec.get("performance", {})
    if not perf.get("required", False):
        print(f"Performance not required for {op_name}")
        return 0

    # Try to run benchmark
    ret, out, err = run_cmd(f"python3 scripts/harness.py bench {op_name}")
    print(out[-1000:] if len(out) > 1000 else out)

    # Write performance report
    op_dir = OPS_DIR / op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    perf_report = {
        "operator": op_name,
        "verified_at": timestamp(),
        "targets": perf.get("targets", []),
        "status": "pending_npu" if ret != 0 else "benchmark_ran",
    }
    save_json(op_dir / "PERF.md", perf_report)

    return 0


def cmd_ship(args):
    """Finalize operator."""
    op_name = args.op_name
    print(f"=== NPU Harness: ship {op_name} ===")
    run_dir = make_run_dir(f"ship_{op_name}")

    checks = {}

    # Check spec exists
    spec_file = SPECS_DIR / f"{op_name}.yaml"
    for f in SPECS_DIR.glob("*.yaml"):
        data = load_yaml(f)
        if data and data.get("name") == op_name:
            spec_file = f
            break
    checks["spec_exists"] = spec_file.exists()

    # Check docs exist
    doc_file = PROJECT_ROOT / "docs" / "operators" / f"{op_name}.md"
    checks["docs_exist"] = doc_file.exists()

    # Check CPU golden exists
    golden_dir = PROJECT_ROOT / "ops" / op_name / "golden"
    checks["cpu_golden_exists"] = golden_dir.exists()

    # Check tests pass
    ret, out, _ = run_cmd(
        f"ctest --test-dir {BUILD_HOST_DIR} -R {op_name} --output-on-failure"
    )
    checks["tests_pass"] = ret == 0

    # Check accuracy report exists
    op_dir = OPS_DIR / op_name
    checks["accuracy_report_exists"] = (op_dir / "RESULTS.json").exists()

    # Check benchmark report
    checks["benchmark_report_exists"] = (op_dir / "PERF.md").exists()

    # Update registry
    if REGISTRY_FILE.exists():
        registry = load_yaml(REGISTRY_FILE)
        if "operators" not in registry:
            registry["operators"] = {}
        registry["operators"][op_name] = registry.get("operators", {}).get(op_name, {})
        registry["operators"][op_name].update({
            "shipped": True,
            "shipped_at": timestamp(),
        })
        save_yaml(REGISTRY_FILE, registry)

    # Report
    report = {
        "operator": op_name,
        "shipped_at": timestamp(),
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    save_json(run_dir / "report.json", report)

    print(f"\nShipping report for {op_name}:")
    for check, passed in checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    print(f"\nOverall: {'READY' if all(checks.values()) else 'INCOMPLETE'}")

    return 0 if all(checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(description="NPU Operator Development Harness CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize Harness state")
    sub.add_parser("map-codebase", help="Analyze and update state")
    sub.add_parser("status", help="Print operator status table")
    sub.add_parser("doctor", help="Diagnose environment")

    p_build = sub.add_parser("build", help="Build project")
    p_build.add_argument("--mode", choices=["host", "npu"], default="host")

    p_test = sub.add_parser("test", help="Run tests")
    p_test.add_argument("--mode", choices=["host", "npu"], default="host")
    p_test.add_argument("--op", help="Operator name filter")

    p_gen = sub.add_parser("generate", help="Generate operator from spec")
    p_gen.add_argument("op_name", help="Operator name")

    p_new = sub.add_parser("new-op", help="Create new operator from spec")
    p_new.add_argument("--spec", required=True, help="Path to spec YAML")

    p_plan = sub.add_parser("plan", help="Create implementation plan")
    p_plan.add_argument("op_name", help="Operator name")

    p_bench = sub.add_parser("bench", help="Run benchmark")
    p_bench.add_argument("op_name", help="Operator name")

    p_verify = sub.add_parser("verify", help="Verify correctness")
    p_verify.add_argument("op_name", help="Operator name")

    p_vperf = sub.add_parser("verify-performance", help="Verify performance")
    p_vperf.add_argument("op_name", help="Operator name")

    p_ship = sub.add_parser("ship", help="Finalize operator")
    p_ship.add_argument("op_name", help="Operator name")

    args = parser.parse_args()

    if args.command == "init":
        return cmd_init(args)
    elif args.command == "map-codebase":
        return cmd_map_codebase(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "build":
        return cmd_build(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "new-op":
        return cmd_generate(args)  # same behavior
    elif args.command == "plan":
        print(f"Plan for {args.op_name}: see .npu-harness/operators/{args.op_name}/PLAN.md")
        return 0
    elif args.command == "bench":
        return cmd_bench(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "verify-performance":
        return cmd_verify_performance(args)
    elif args.command == "ship":
        return cmd_ship(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
