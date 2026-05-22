#include "harness/accuracy.hpp"
// accuracy.hpp is header-only; this file exists for CMake library completeness
namespace harness {
namespace accuracy {
// Template instantiations for common types
template AccuracyReport CompareArrays<float>(const float*, const float*, size_t, Tolerance, size_t);
template AccuracyReport CompareArrays<double>(const double*, const double*, size_t, Tolerance, size_t);
template AccuracyReport CompareArrays<int32_t>(const int32_t*, const int32_t*, size_t, Tolerance, size_t);
template AccuracyReport CompareArrays<uint16_t>(const uint16_t*, const uint16_t*, size_t, Tolerance, size_t);
}  // namespace accuracy
}  // namespace harness
