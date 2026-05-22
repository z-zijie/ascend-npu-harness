set(GOLDEN_SOURCES golden/reduce_sum_lastdim_golden.hpp golden/reduce_sum_lastdim_golden.cpp)
set(HOST_SOURCES host/reduce_sum_lastdim_tiling.hpp host/reduce_sum_lastdim_tiling.cpp)
set(RUNNER_SOURCES runner/reduce_sum_lastdim_runner.hpp runner/reduce_sum_lastdim_runner.cpp)
set(KERNEL_SOURCES kernel/reduce_sum_lastdim_kernel.cpp)
harness_add_operator(reduce_sum_lastdim HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_reduce_sum_lastdim.cpp)
