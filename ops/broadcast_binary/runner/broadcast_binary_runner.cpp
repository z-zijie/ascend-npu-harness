#include "broadcast_binary_runner.hpp"
#include "broadcast_binary_golden.hpp"

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
#include "harness/ascend_runtime.hpp"
#include "broadcast_binary_tiling.hpp"
#endif

namespace harness {
namespace runner {
namespace broadcast_binary {

void run(const float* x, const float* y, const float* bias, float* z,
         int64_t B, int64_t C, int64_t H, int64_t W, int64_t y_batch) {

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
    if (harness::IsAscendRuntimeAvailable() && harness::HasUsableDevice()) {
        size_t n_x = static_cast<size_t>(B * C * H * W);
        size_t n_y = static_cast<size_t>(y_batch * C);
        size_t n_bias = static_cast<size_t>(C);
        size_t n_z = n_x;

        auto tiling = tiling::broadcast_binary::compute_tiling(
            static_cast<uint32_t>(B), static_cast<uint32_t>(C),
            static_cast<uint32_t>(H), static_cast<uint32_t>(W),
            static_cast<uint32_t>(y_batch));

        harness::ascend::DeviceMemory d_x(n_x * sizeof(float));
        harness::ascend::DeviceMemory d_y(n_y * sizeof(float));
        harness::ascend::DeviceMemory d_bias(n_bias * sizeof(float));
        harness::ascend::DeviceMemory d_z(n_z * sizeof(float));

        harness::ascend::MemcpyHostToDevice(d_x.data(), x, n_x * sizeof(float));
        harness::ascend::MemcpyHostToDevice(d_y.data(), y, n_y * sizeof(float));
        harness::ascend::MemcpyHostToDevice(d_bias.data(), bias, n_bias * sizeof(float));

        // Kernel launch would go here in full CANN build

        harness::ascend::MemcpyDeviceToHost(z, d_z.data(), n_z * sizeof(float));
        return;
    }
#endif

    golden::broadcast_binary::compute(x, y, bias, z, B, C, H, W, y_batch);
}

}  // namespace broadcast_binary
}  // namespace runner
}  // namespace harness
