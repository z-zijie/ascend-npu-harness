cmake_minimum_required(VERSION 3.16)

# Use CMAKE_PREFIX_PATH to find ASC package
list(APPEND CMAKE_PREFIX_PATH "/home/developer/Ascend/cann-9.0.0/compiler/tikcpp/ascendc_kernel_cmake")
find_package(ASC REQUIRED)

project(elementwise_add_prof LANGUAGES ASC CXX)

add_executable(elementwise_add_prof
    elementwise_add.asc
)

target_compile_options(elementwise_add_prof PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

target_include_directories(elementwise_add_prof PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
