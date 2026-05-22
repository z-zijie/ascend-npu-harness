set(GOLDEN_SOURCES golden/transpose_2d_4d_golden.hpp golden/transpose_2d_4d_golden.cpp)
set(HOST_SOURCES host/transpose_2d_4d_tiling.hpp host/transpose_2d_4d_tiling.cpp)
set(RUNNER_SOURCES runner/transpose_2d_4d_runner.hpp runner/transpose_2d_4d_runner.cpp)
set(KERNEL_SOURCES kernel/transpose_2d_4d_kernel.cpp)
harness_add_operator(transpose_2d_4d HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_transpose_2d_4d.cpp)
