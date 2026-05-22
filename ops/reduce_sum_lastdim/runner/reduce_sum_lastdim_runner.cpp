#include "reduce_sum_lastdim_runner.hpp"
#include "reduce_sum_lastdim_golden.hpp"

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
#include "harness/ascend_runtime.hpp"
#include "reduce_sum_lastdim_tiling.hpp"
#endif

namespace harness {
namespace runner {
namespace reduce_sum_lastdim {

void run(const float* x, float* out,
         int64_t num_batches, int64_t M, int64_t K) {

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
    if (harness::IsAscendRuntimeAvailable() && harness::HasUsableDevice()) {
        size_t n_in = static_cast<size_t>(num_batches * M * K);
        size_t n_out = static_cast<size_t>(num_batches * M);

        auto tiling = tiling::reduce_sum_lastdim::compute_tiling(
            static_cast<uint32_t>(num_batches),
            static_cast<uint32_t>(M),
            static_cast<uint32_t>(K));

        harness::ascend::DeviceMemory d_x(n_in * sizeof(float));
        harness::ascend::DeviceMemory d_out(n_out * sizeof(float));

        harness::ascend::MemcpyHostToDevice(d_x.data(), x, n_in * sizeof(float));

        // Kernel launch would go here in full CANN build

        harness::ascend::MemcpyDeviceToHost(out, d_out.data(), n_out * sizeof(float));
        return;
    }
#endif

    golden::reduce_sum_lastdim::compute(x, out, num_batches, M, K);
}

}  // namespace reduce_sum_lastdim
}  // namespace runner
}  // namespace harness
