# Troubleshooting

## Common Issues

### CMake: GTest not found
```
sudo apt install libgtest-dev
```
Or set `HARNESS_ENABLE_TESTS=OFF` to skip tests.

### CMake: Ninja not found
```
pip3 install ninja
```
Or use Make: add `-G "Unix Makefiles"` to cmake command.

### NPU: No devices visible
Check driver installation:
```bash
ls /dev/davinci*
npu-smi info
```
If no devices: enable host-only mode with `-DHARNESS_ENABLE_ASCEND=OFF`.

### Build: Ascend C compilation errors
- Verify CANN version with: `cat $ASCEND_HOME_PATH/version.cfg`
- Check ccec in PATH: `which ccec`
- Run environment detection: `python3 scripts/detect_ascend_env.py`

### Tests: GTEST_SKIP messages
Normal in host-only mode. NPU tests skip when device unavailable.
To run NPU tests: ensure `HARNESS_ENABLE_ASCEND=ON` and device visible.

### Accuracy failures
1. Check tolerance settings in operator spec
2. Verify CPU golden implementation
3. For fp16: expect larger tolerances (1e-2 elementwise, 5e-2 reduction)
4. Check for numerical stability issues (use float32 accumulation)

### Performance gaps
1. Verify benchmark is measuring the correct path (NPU, not CPU fallback)
2. Check tiling configuration
3. Run `python3 scripts/harness.py doctor` for diagnostics

## Quick Diagnostics

```bash
# Environment check
python3 scripts/detect_ascend_env.py

# Full doctor
python3 scripts/harness.py doctor

# Build status
python3 scripts/harness.py status

# Rebuild from scratch
rm -rf build/
cmake -S . -B build/host -G Ninja -DHARNESS_ENABLE_ASCEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/host
```

## Getting Help

- Check operator status: `/npu-status`
- Run doctor: `/npu-doctor`
- Check operator decisions: `.npu-harness/operators/<op>/DECISIONS.md`
- Check operator blockers: `.npu-harness/operators/<op>/BLOCKERS.md`
