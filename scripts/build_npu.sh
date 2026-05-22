#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== NPU Harness: NPU Build ==="
echo "Target arch: dav-2201"

cmake -S . -B build/npu -G Ninja \
  -DHARNESS_ENABLE_ASCEND=ON \
  -DHARNESS_NPU_ARCH=dav-2201 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build/npu
echo "NPU build complete."
