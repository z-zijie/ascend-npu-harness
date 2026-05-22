#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
OP="${1:?Usage: $0 <op_name>}"
python3 scripts/harness.py bench "$OP"
