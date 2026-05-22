set(GOLDEN_SOURCES golden/layer_norm_golden.hpp golden/layer_norm_golden.cpp)
set(HOST_SOURCES host/layer_norm_tiling.hpp host/layer_norm_tiling.cpp)
set(RUNNER_SOURCES runner/layer_norm_runner.hpp runner/layer_norm_runner.cpp)
set(KERNEL_SOURCES kernel/layer_norm_kernel.cpp)
harness_add_operator(layer_norm HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_layer_norm.cpp)
