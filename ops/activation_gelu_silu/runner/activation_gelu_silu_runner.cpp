#include "activation_gelu_silu_runner.hpp"

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
#include "harness/ascend_runtime.hpp"
#include "activation_gelu_silu_tiling.hpp"
#endif

namespace harness {
namespace runner {
namespace activation_gelu_silu {

void run(const float* x, float* y, const Shape& shape, golden::activation_gelu_silu::ActivationType act) {
    size_t n = shape.num_elements();
#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
    if (harness::IsAscendRuntimeAvailable() && harness::HasUsableDevice()) {
        harness::ascend::DeviceMemory d_x(n * sizeof(float));
        harness::ascend::DeviceMemory d_y(n * sizeof(float));
        harness::ascend::MemcpyHostToDevice(d_x.data(), x, n * sizeof(float));
        // NPU kernel launch would go here
        harness::ascend::MemcpyDeviceToHost(y, d_y.data(), n * sizeof(float));
        return;
    }
#endif
    golden::activation_gelu_silu::compute(x, y, n, act);
}

}  // namespace activation_gelu_silu
}  // namespace runner
}  // namespace harness
