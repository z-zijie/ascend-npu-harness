#include "transpose_2d_4d_runner.hpp"

#if defined(HARNESS_HAS_ASCEND) && HARNESS_HAS_ASCEND == 1
#include "ascend_runtime.hpp"
#endif

namespace harness {
namespace transpose_2d_4d {

// Runner is header-only template; this .cpp exists for CMake completeness
// and for any non-template runner code

}  // namespace transpose_2d_4d
}  // namespace harness
