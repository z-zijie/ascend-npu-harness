# Build Instructions

## Prerequisites

- CMake >= 3.20
- Ninja
- G++ >= 9 (C++17)
- Google Test (libgtest-dev)
- Python 3.8+ with PyYAML

### Optional: CANN/NPU
- Ascend CANN toolkit (for NPU mode)
- Ascend NPU device (for NPU tests)

## Host-Only Build

```bash
# Configure
cmake -S . -B build/host -G Ninja \
  -DHARNESS_ENABLE_ASCEND=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build
cmake --build build/host

# Test
ctest --test-dir build/host --output-on-failure

# Run specific test
ctest --test-dir build/host -R elementwise_add

# Run by label
ctest --test-dir build/host -L golden
```

## NPU Build (with Ascend)

```bash
# Detect environment first
python3 scripts/detect_ascend_env.py

# Configure
cmake -S . -B build/npu -G Ninja \
  -DHARNESS_ENABLE_ASCEND=ON \
  -DHARNESS_NPU_ARCH=dav-2201 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build
cmake --build build/npu

# Test
ctest --test-dir build/npu --output-on-failure
```

## Harness CLI

```bash
# Initialize
python3 scripts/harness.py init

# Map codebase
python3 scripts/harness.py map-codebase

# Status
python3 scripts/harness.py status

# Doctor
python3 scripts/harness.py doctor

# Build
python3 scripts/harness.py build --mode host
python3 scripts/harness.py build --mode npu

# Test
python3 scripts/harness.py test --mode host --op elementwise_add
```

## CMake Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| HARNESS_ENABLE_ASCEND | BOOL | OFF | Enable Ascend C/NPU build |
| HARNESS_NPU_ARCH | STRING | dav-2201 | Target NPU architecture |
| HARNESS_ENABLE_TESTS | BOOL | ON | Enable tests |
| HARNESS_ENABLE_BENCHMARKS | BOOL | ON | Enable benchmarks |
| HARNESS_ENABLE_WARNINGS | BOOL | ON | Enable compiler warnings |
| HARNESS_STRICT | BOOL | OFF | Treat warnings as errors |
| ASCEND_HOME | PATH | auto | Override Ascend install path |
