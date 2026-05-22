set(GOLDEN_SOURCES golden/activation_gelu_silu_golden.hpp golden/activation_gelu_silu_golden.cpp)
set(HOST_SOURCES host/activation_gelu_silu_tiling.hpp host/activation_gelu_silu_tiling.cpp)
set(RUNNER_SOURCES runner/activation_gelu_silu_runner.hpp runner/activation_gelu_silu_runner.cpp)
set(KERNEL_SOURCES kernel/activation_gelu_silu_kernel.cpp)
harness_add_operator(activation_gelu_silu HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_activation_gelu_silu.cpp)
