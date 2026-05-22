# ROADMAP.md — Harness Development Roadmap

## Phase 1: Foundation (Current)
- [x] Directory structure
- [ ] CLAUDE.md and AGENTS.md
- [ ] State files (PROJECT, REQUIREMENTS, ROADMAP, STATE, CONTEXT)
- [ ] CMake build system (host + NPU modes)
- [ ] Core C++ utilities (accuracy, shape, dtype, runtime, logging)
- [ ] Python scripts (harness.py, detect_ascend_env.py, spec generator)
- [ ] Operator registry

## Phase 2: Agentic Layer
- [ ] Skills (10 skills under .claude/skills/)
- [ ] Slash commands (11 commands under .claude/commands/)
- [ ] .npu-harness templates and state
- [ ] Harness CLI full implementation

## Phase 3: Operator Bootstrap (1-5)
- [ ] elementwise_add
- [ ] axpy_scalar_affine
- [ ] broadcast_binary
- [ ] activation_gelu_silu
- [ ] reduce_sum_lastdim

## Phase 4: Operator Bootstrap (6-10)
- [ ] transpose_2d_4d
- [ ] layer_norm
- [ ] matmul_bias
- [ ] rms_norm_rope_fused
- [ ] flash_attention_fwd_fused

## Phase 5: Verification
- [ ] Host build and test
- [ ] NPU build and test (if available)
- [ ] Per-operator status report
- [ ] Final report
