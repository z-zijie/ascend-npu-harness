set(GOLDEN_SOURCES golden/axpy_scalar_affine_golden.hpp golden/axpy_scalar_affine_golden.cpp)
set(HOST_SOURCES host/axpy_scalar_affine_tiling.hpp host/axpy_scalar_affine_tiling.cpp)
set(RUNNER_SOURCES runner/axpy_scalar_affine_runner.hpp runner/axpy_scalar_affine_runner.cpp)
set(KERNEL_SOURCES kernel/axpy_scalar_affine_kernel.cpp)
harness_add_operator(axpy_scalar_affine HAS_KERNEL HAS_BENCHMARK
    GOLDEN_SOURCES ${GOLDEN_SOURCES} HOST_SOURCES ${HOST_SOURCES}
    RUNNER_SOURCES ${RUNNER_SOURCES} KERNEL_SOURCES ${KERNEL_SOURCES}
    BENCH_SOURCES bench/bench_axpy_scalar_affine.cpp)
