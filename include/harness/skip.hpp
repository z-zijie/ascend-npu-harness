#pragma once
#include <string>

// This header provides skip helpers for GTest.
// When GTest is available, GTEST_SKIP() is used.
// When not, tests are compiled with stubs.

#ifdef HARNESS_HAS_GTEST
#include <gtest/gtest.h>
#define HARNESS_GTEST_SKIP(msg) GTEST_SKIP() << msg
#else
#define HARNESS_GTEST_SKIP(msg) do { \
    std::cerr << "SKIP: " << msg << std::endl; \
    return; \
} while(0)
#endif

namespace harness {

// Check if Ascend runtime is available
bool IsAscendRuntimeAvailable();

// Check if a usable NPU device is present
bool HasUsableDevice();

// Get the number of available devices
int GetDeviceCount();

}  // namespace harness
