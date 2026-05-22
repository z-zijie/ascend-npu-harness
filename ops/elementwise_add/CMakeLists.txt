set(GOLDEN_SOURCES golden/elementwise_add_golden.hpp golden/elementwise_add_golden.cpp)
set(HOST_SOURCES host/elementwise_add_tiling.hpp host/elementwise_add_tiling.cpp)
set(RUNNER_SOURCES runner/elementwise_add_runner.hpp runner/elementwise_add_runner.cpp)
set(KERNEL_SOURCES kernel/elementwise_add_kernel.cpp)
harness_add_operator(elementwise_add HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_elementwise_add.cpp)
