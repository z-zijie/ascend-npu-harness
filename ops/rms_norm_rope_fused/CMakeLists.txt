set(GOLDEN_SOURCES golden/rms_norm_rope_fused_golden.hpp golden/rms_norm_rope_fused_golden.cpp)
set(HOST_SOURCES host/rms_norm_rope_fused_tiling.hpp host/rms_norm_rope_fused_tiling.cpp)
set(RUNNER_SOURCES runner/rms_norm_rope_fused_runner.hpp runner/rms_norm_rope_fused_runner.cpp)
set(KERNEL_SOURCES kernel/rms_norm_rope_fused_kernel.cpp)
harness_add_operator(rms_norm_rope_fused HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_rms_norm_rope_fused.cpp)
