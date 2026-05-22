#!/usr/bin/env python3
"""Create a new operator from scratch — generates spec and skeleton files."""

import sys
import yaml
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"


def create_minimal_spec(op_name, category="elementwise"):
    return {
        "schema_version": 1,
        "name": op_name,
        "category": category,
        "target_arch": "dav-2201",
        "summary": f"{op_name} operator",
        "math": {
            "definition": "",
            "equations": [],
            "notes": []
        },
        "inputs": [
            {
                "name": "x",
                "dtype": ["float32", "float16"],
                "shape": ["N"],
                "layout": "contiguous",
                "role": "activation"
            }
        ],
        "outputs": [
            {
                "name": "y",
                "dtype": ["float32", "float16"],
                "shape": ["N"],
                "layout": "contiguous"
            }
        ],
        "parameters": [],
        "shape_symbols": {
            "N": {"allowed": [1, 8, 64, 128, 1024, 4096]}
        },
        "shape_rules": [],
        "dtype_rules": {
            "accumulation": "float32",
            "output_cast": "same_as_input"
        },
        "layout_rules": {
            "contiguous_required": True,
            "supported_layouts": ["ND"]
        },
        "broadcasting": {"supported": False, "rules": []},
        "masking": {"supported": False, "types": []},
        "tiling": {
            "strategy": "auto",
            "constraints": [],
            "workspace": "optional"
        },
        "implementation": {
            "cpu_golden": "required",
            "ascend_kernel": "required",
            "direct_launch": "required",
            "framework_substitution_allowed": False
        },
        "accuracy": {
            "float32": {"abs": 1e-5, "rel": 1e-5},
            "float16": {"abs": 1e-2, "rel": 1e-2}
        },
        "performance": {
            "required": False,
            "targets": [],
            "tuning_budget": {"max_iterations": 5}
        },
        "tests": {
            "deterministic_seed": 20260522,
            "shapes": [[1024], [1], [8]],
            "edge_cases": [],
            "invalid_cases": []
        },
        "benchmark": {
            "required": False,
            "warmup": 10,
            "iterations": 100
        },
        "status": {
            "generated": False,
            "built": False,
            "host_tested": False,
            "npu_tested": False,
            "accuracy_passed": False,
            "performance_passed": False
        }
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 new_op.py <op_name> [category]")
        print("  category: elementwise, reduction, layout, matmul, normalization, attention, fusion, custom")
        sys.exit(1)

    op_name = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "elementwise"

    spec = create_minimal_spec(op_name, category)

    spec_path = SPECS_DIR / f"{op_name}.yaml"
    os.makedirs(SPECS_DIR, exist_ok=True)

    with open(spec_path, 'w') as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    print(f"Created spec: {spec_path}")
    print(f"\nNext: python3 scripts/generate_op_from_spec.py {spec_path}")
    print(f"  Then edit the spec for your operator's requirements.")
    print(f"  Then re-run the generator to create skeleton files.")


if __name__ == "__main__":
    main()
