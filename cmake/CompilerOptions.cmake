# CompilerOptions.cmake — Compiler flags and architecture-specific options

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Detected aarch64 host")
    add_compile_definitions(HARNESS_HOST_AARCH64=1)
else()
    message(STATUS "Host architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
    add_compile_definitions(HARNESS_DEBUG=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-g -O2)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
endif()

# Position-independent code (needed for shared library wrapping)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Math library
find_library(MATH_LIB m)
if(MATH_LIB)
    link_libraries(${MATH_LIB})
endif()
