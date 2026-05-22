# CONTEXT.md — Harness Operational Context

## What This Harness Is

A domain-specific agentic engineering system for Ascend NPU operator development. It combines:
- A spec-driven workflow layer (skills, commands, state files)
- A concrete C++/Ascend C engineering layer (CMake, GTest, kernels)
- Designed for repeatable, evidence-based operator development

## What This Harness Is Not

- A PyTorch extension library (though operators can be wrapped for PyTorch)
- A general-purpose ML compiler
- A replacement for CANN toolchain
- A framework operator substitution system

## Key Constraints

1. **Default NPU arch is dav-2201.** All generated code targets this arch.
2. **Direct `<<<>>>` launch is mandatory.** No framework op substitution.
3. **CPU golden is mandatory.** Every operator must have a C++ reference.
4. **Host-only mode must always work.** CANN/NPU are optional for basic development.
5. **State survives context resets.** The filesystem is the source of truth.

## Working With This Harness

### Starting a new operator
```bash
/npu-new-op <<<description of the operator>>>
```

### Checking status
```bash
/npu-status
python3 scripts/harness.py status
```

### Building and testing
```bash
python3 scripts/harness.py build --mode host
python3 scripts/harness.py test --mode host
python3 scripts/harness.py build --mode npu  # if CANN available
python3 scripts/harness.py test --mode npu   # if NPU available
```

### Environment detection
```bash
python3 scripts/detect_ascend_env.py
```

## CANN Environment (detected)

- CANN version: 9.0.0
- Install path: /home/developer/Ascend/cann-9.0.0
- Compiler: ccec (in PATH)
- NPU devices: 2x Ascend910 (davinci 6, 7)
- Driver: installed (/usr/local/Ascend/driver)

## Repository Layout

```
.
├── CLAUDE.md              # Harness agent instructions
├── AGENTS.md              # Agent role definitions
├── PROJECT.md             # Project overview
├── REQUIREMENTS.md        # Functional/non-functional requirements
├── ROADMAP.md             # Development roadmap
├── STATE.md               # Current state snapshot
├── CONTEXT.md             # This file
├── OPERATOR_REGISTRY.yaml # Operator inventory
├── CMakeLists.txt         # Root CMake
├── CMakePresets.json      # CMake presets
├── cmake/                 # CMake modules
├── include/harness/       # C++ utility headers
├── src/                   # C++ utility implementations
├── specs/                 # Operator spec YAML files
├── ops/                   # Operator implementations
├── tests/                 # GTest test files
├── scripts/               # Python/Shell scripts
├── docs/                  # Documentation
├── .claude/               # Claude Code integration
│   ├── commands/          # Slash command definitions
│   └── skills/            # Skill definitions
└── .npu-harness/          # Harness persistent state
    ├── config.yaml
    ├── state/             # State file copies
    ├── operators/         # Per-operator state
    ├── runs/              # Timestamped run logs
    └── templates/         # Document templates
```
