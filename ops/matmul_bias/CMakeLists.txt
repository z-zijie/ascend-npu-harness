set(GOLDEN_SOURCES golden/matmul_bias_golden.hpp golden/matmul_bias_golden.cpp)
set(HOST_SOURCES host/matmul_bias_tiling.hpp host/matmul_bias_tiling.cpp)
set(RUNNER_SOURCES runner/matmul_bias_runner.hpp runner/matmul_bias_runner.cpp)
set(KERNEL_SOURCES kernel/matmul_bias_kernel.cpp)
harness_add_operator(matmul_bias HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_matmul_bias.cpp)
