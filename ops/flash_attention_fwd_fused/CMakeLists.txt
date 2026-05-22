set(GOLDEN_SOURCES golden/flash_attention_fwd_fused_golden.hpp golden/flash_attention_fwd_fused_golden.cpp)
set(HOST_SOURCES host/flash_attention_fwd_fused_tiling.hpp host/flash_attention_fwd_fused_tiling.cpp)
set(RUNNER_SOURCES runner/flash_attention_fwd_fused_runner.hpp runner/flash_attention_fwd_fused_runner.cpp)
set(KERNEL_SOURCES kernel/flash_attention_fwd_fused_kernel.cpp)
harness_add_operator(flash_attention_fwd_fused HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_flash_attention_fwd_fused.cpp)
