cmake_minimum_required(VERSION 3.20)
project(npu-harness VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Harness options
# ============================================================================
option(HARNESS_ENABLE_ASCEND "Enable Ascend C/NPU build" OFF)
set(HARNESS_NPU_ARCH "dav-2201" CACHE STRING "Target NPU architecture")
option(HARNESS_ENABLE_TESTS "Enable tests" ON)
option(HARNESS_ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(HARNESS_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(HARNESS_STRICT "Treat warnings as errors" OFF)

# ============================================================================
# dav-2201 enforcement
# ============================================================================
if(HARNESS_ENABLE_ASCEND)
    if(HARNESS_NPU_ARCH STREQUAL "")
        set(HARNESS_NPU_ARCH "dav-2201")
        message(STATUS "HARNESS_NPU_ARCH was empty; defaulting to dav-2201")
    endif()
    message(STATUS "NPU target architecture: ${HARNESS_NPU_ARCH}")
    if(NOT HARNESS_NPU_ARCH STREQUAL "dav-2201")
        message(WARNING "Target arch is ${HARNESS_NPU_ARCH}. The default for this project is dav-2201.")
    endif()
endif()

# ============================================================================
# C++ standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compiler options
# ============================================================================
if(HARNESS_ENABLE_WARNINGS)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
if(HARNESS_STRICT)
    add_compile_options(-Werror)
endif()

# ============================================================================
# Include cmake modules
# ============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(Ascend)
include(CompilerOptions)
include(GTest)
include(OperatorTarget)

# ============================================================================
# Harness core library (always built, no Ascend dependency)
# ============================================================================
add_library(harness_core STATIC
    src/accuracy.cpp
    src/dtype.cpp
    src/shape.cpp
    src/test_utils.cpp
)
target_include_directories(harness_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(harness_core PUBLIC
    ${CMAKE_DL_LIBS}
)

# Ascend runtime (only when Ascend enabled)
if(HARNESS_ENABLE_ASCEND)
    target_sources(harness_core PRIVATE src/ascend_runtime.cpp)
    target_compile_definitions(harness_core PUBLIC HARNESS_HAS_ASCEND=1)
    target_compile_definitions(harness_core PUBLIC HARNESS_NPU_ARCH_DEFAULT="${HARNESS_NPU_ARCH}")

    # Link Ascend libraries
    harness_find_ascend()
    if(ASCEND_FOUND)
        target_include_directories(harness_core PUBLIC ${ASCEND_INCLUDE_DIRS})
        target_link_directories(harness_core PUBLIC ${ASCEND_LIBRARY_DIRS})
        target_link_libraries(harness_core PUBLIC ${ASCEND_LIBRARIES})
        message(STATUS "Ascend runtime enabled")
    else()
        message(WARNING "Ascend libraries not found; NPU runtime will be stubbed")
        target_compile_definitions(harness_core PUBLIC HARNESS_ASCEND_STUB=1)
    endif()
else()
    target_sources(harness_core PRIVATE src/ascend_runtime.cpp)
    target_compile_definitions(harness_core PUBLIC HARNESS_HAS_ASCEND=0)
    target_compile_definitions(harness_core PUBLIC HARNESS_NPU_ARCH_DEFAULT="${HARNESS_NPU_ARCH}")
    message(STATUS "Host-only mode; Ascend runtime stubbed")
endif()

# ============================================================================
# Operator subdirectories
# ============================================================================
add_subdirectory(ops)

# ============================================================================
# Tests
# ============================================================================
if(HARNESS_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Print configuration summary
# ============================================================================
message(STATUS "")
message(STATUS "=== NPU Harness Configuration ===")
message(STATUS "  HARNESS_ENABLE_ASCEND:    ${HARNESS_ENABLE_ASCEND}")
message(STATUS "  HARNESS_NPU_ARCH:         ${HARNESS_NPU_ARCH}")
message(STATUS "  HARNESS_ENABLE_TESTS:     ${HARNESS_ENABLE_TESTS}")
message(STATUS "  HARNESS_ENABLE_BENCHMARKS: ${HARNESS_ENABLE_BENCHMARKS}")
message(STATUS "  Build type:               ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:             ${CMAKE_CXX_STANDARD}")
message(STATUS "=================================")
message(STATUS "")
