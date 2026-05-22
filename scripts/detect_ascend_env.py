#!/usr/bin/env python3
"""Detect CANN/Ascend environment and report status."""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def find_file(pattern, search_dirs):
    for d in search_dirs:
        for root, dirs, files in os.walk(d):
            for f in files:
                if pattern in f:
                    return os.path.join(root, f)
    return None


def check_path(path):
    return os.path.exists(path)


def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode, result.stdout.strip() + result.stderr.strip()
    except Exception as e:
        return -1, str(e)


def detect():
    print("=" * 60)
    print("NPU Harness — Ascend Environment Detection")
    print("=" * 60)

    # CANN installation
    cann_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or "/usr/local/Ascend"

    print(f"\n--- CANN Toolkit ---")
    if cann_home and check_path(cann_home):
        print(f"  CANN_HOME: {cann_home} [FOUND]")
    else:
        print(f"  CANN_HOME: not found")

    # Check known paths
    known_paths = [
        cann_home,
        "/home/developer/Ascend/cann-9.0.0",
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend/ascend-toolkit",
    ]
    found_cann = None
    for p in known_paths:
        if p and check_path(p):
            found_cann = p
            break

    if found_cann:
        print(f"  Detected CANN: {found_cann}")

        # Include paths
        inc_candidates = [
            os.path.join(found_cann, "include"),
            os.path.join(found_cann, "compiler", "include"),
            os.path.join(found_cann, "toolkit", "include"),
        ]
        for inc in inc_candidates:
            if check_path(inc):
                print(f"  Include: {inc}")

        # Library paths
        lib_candidates = [
            os.path.join(found_cann, "lib64"),
            os.path.join(found_cann, "compiler", "lib64"),
        ]
        for lib in lib_candidates:
            if check_path(lib):
                print(f"  Library: {lib}")
    else:
        print("  CANN toolkit not detected.")
        print("  Host-only mode is available.")
        print("  Use -DHARNESS_ENABLE_ASCEND=OFF for CPU golden and GTest validation.")

    # Compiler
    print(f"\n--- Compiler ---")
    ccec = shutil.which("ccec")
    if ccec:
        print(f"  ccec: {ccec} [FOUND]")
    else:
        print(f"  ccec: not in PATH")

    gcc = shutil.which("g++")
    if gcc:
        ret, ver = run_cmd("g++ --version 2>&1 | head -1")
        print(f"  g++: {gcc} ({ver})")

    # Build tools
    print(f"\n--- Build Tools ---")
    for tool in ["cmake", "ninja", "make"]:
        p = shutil.which(tool)
        if p:
            ret, ver = run_cmd(f"{tool} --version 2>&1 | head -1")
            print(f"  {tool}: {p} ({ver})")
        else:
            print(f"  {tool}: NOT FOUND")

    # GTest
    print(f"\n--- GTest ---")
    gtest_h = find_file("gtest.h", ["/usr/include", "/usr/local/include"])
    if gtest_h:
        print(f"  GTest headers: {gtest_h} [FOUND]")
    else:
        print(f"  GTest headers: NOT FOUND")

    gtest_lib = find_file("libgtest", ["/usr/lib", "/usr/local/lib"])
    if gtest_lib:
        print(f"  GTest libs: {gtest_lib} [FOUND]")
    else:
        print(f"  GTest libs: NOT FOUND")

    # NPU devices
    print(f"\n--- NPU Devices ---")
    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        ret, out = run_cmd("npu-smi info -t npu -c 0 2>&1 | head -20")
        if ret == 0 and out:
            print(f"  npu-smi: {npu_smi} [FOUND]")
            print(f"  {out[:500]}")
    else:
        print(f"  npu-smi: not in PATH")

    # Device nodes
    davinci_devices = list(Path("/dev").glob("davinci*"))
    if davinci_devices:
        print(f"  Device nodes: {len(davinci_devices)} found")
        for d in davinci_devices[:8]:
            print(f"    {d}")
    else:
        print(f"  No /dev/davinci* devices found")

    # Driver
    print(f"\n--- Driver ---")
    driver_dir = "/usr/local/Ascend/driver"
    if check_path(driver_dir):
        print(f"  Driver directory: {driver_dir} [FOUND]")
        lib64 = os.path.join(driver_dir, "lib64")
        if check_path(lib64):
            libs = [f for f in os.listdir(lib64) if f.endswith('.so')]
            print(f"  Driver libs: {len(libs)} found")
    else:
        print(f"  Driver: not found at {driver_dir}")

    # Environment variables
    print(f"\n--- Environment Variables ---")
    ascend_vars = [k for k in os.environ if 'ASCEND' in k or 'CANN' in k or 'NPU' in k]
    for v in sorted(ascend_vars):
        print(f"  {v}={os.environ[v][:100]}")

    # Recommendation
    print(f"\n--- Recommended CMake Commands ---")
    if found_cann:
        print(f"  # Host-only (always works):")
        print(f"  cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo")
        print(f"  cmake --build build/host")
        print(f"  ctest --test-dir build/host --output-on-failure")
        print(f"")
        print(f"  # NPU mode:")
        print(f"  cmake -S . -B build/npu -G Ninja -DHARNESS_ENABLE_ASCEND=ON -DHARNESS_NPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=RelWithDebInfo")
        print(f"  cmake --build build/npu")
        print(f"  ctest --test-dir build/npu --output-on-failure")
    else:
        print(f"  # Host-only (CANN not available):")
        print(f"  cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo")
        print(f"  cmake --build build/host")
        print(f"  ctest --test-dir build/host --output-on-failure")

    print(f"\n{'=' * 60}")
    print(f"Detection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(detect())
