#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
OP="${1:-}"
if [ -n "$OP" ]; then
    ctest --test-dir build/host -R "$OP" --output-on-failure
else
    ctest --test-dir build/host --output-on-failure
fi
