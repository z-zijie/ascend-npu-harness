#include "axpy_scalar_affine_runner.hpp"
#include "axpy_scalar_affine_golden.hpp"

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
#include "harness/ascend_runtime.hpp"
#include "axpy_scalar_affine_tiling.hpp"
#endif

namespace harness {
namespace runner {
namespace axpy_scalar_affine {

void run(const float* x, float* y, const Shape& shape, float alpha, float beta) {
    size_t n = shape.num_elements();

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
    if (harness::IsAscendRuntimeAvailable() && harness::HasUsableDevice()) {
        auto tiling = tiling::axpy_scalar_affine::compute_tiling(
            static_cast<uint32_t>(n), alpha, beta);

        harness::ascend::DeviceMemory d_x(n * sizeof(float));
        harness::ascend::DeviceMemory d_y(n * sizeof(float));

        harness::ascend::MemcpyHostToDevice(d_x.data(), x, n * sizeof(float));

        // Kernel launch would go here in full CANN build

        harness::ascend::MemcpyDeviceToHost(y, d_y.data(), n * sizeof(float));
        return;
    }
#endif

    golden::axpy_scalar_affine::compute(x, y, n, alpha, beta);
}

}  // namespace axpy_scalar_affine
}  // namespace runner
}  // namespace harness
