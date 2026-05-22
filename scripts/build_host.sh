#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== NPU Harness: Host Build ==="
cmake -S . -B build/host -G Ninja \
  -DHARNESS_ENABLE_ASCEND=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build/host
echo "Build complete."
