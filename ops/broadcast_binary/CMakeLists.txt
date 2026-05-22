set(GOLDEN_SOURCES golden/broadcast_binary_golden.hpp golden/broadcast_binary_golden.cpp)
set(HOST_SOURCES host/broadcast_binary_tiling.hpp host/broadcast_binary_tiling.cpp)
set(RUNNER_SOURCES runner/broadcast_binary_runner.hpp runner/broadcast_binary_runner.cpp)
set(KERNEL_SOURCES kernel/broadcast_binary_kernel.cpp)
harness_add_operator(broadcast_binary HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_broadcast_binary.cpp)
