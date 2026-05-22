# OperatorTarget.cmake — Helper function to create operator library targets

function(harness_add_operator OP_NAME)
    set(options HAS_KERNEL HAS_BENCHMARK)
    set(oneValueArgs "")
    set(multiValueArgs KERNEL_SOURCES HOST_SOURCES GOLDEN_SOURCES RUNNER_SOURCES BENCH_SOURCES)
    cmake_parse_arguments(OP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(OP_DIR "${CMAKE_SOURCE_DIR}/ops/${OP_NAME}")

    # Collect source files
    set(_all_sources "")

    # Golden sources (required)
    if(OP_GOLDEN_SOURCES)
        set(_golden_files "")
        foreach(s ${OP_GOLDEN_SOURCES})
            list(APPEND _golden_files "${OP_DIR}/${s}")
        endforeach()
        list(APPEND _all_sources ${_golden_files})
    endif()

    # Host tiling sources (required)
    if(OP_HOST_SOURCES)
        set(_host_files "")
        foreach(s ${OP_HOST_SOURCES})
            list(APPEND _host_files "${OP_DIR}/${s}")
        endforeach()
        list(APPEND _all_sources ${_host_files})
    endif()

    # Runner sources
    if(OP_RUNNER_SOURCES)
        set(_runner_files "")
        foreach(s ${OP_RUNNER_SOURCES})
            list(APPEND _runner_files "${OP_DIR}/${s}")
        endforeach()
        list(APPEND _all_sources ${_runner_files})
    endif()

    # Kernel sources (only when Ascend enabled)
    if(HARNESS_ENABLE_ASCEND AND OP_HAS_KERNEL AND OP_KERNEL_SOURCES)
        set(_kernel_files "")
        foreach(s ${OP_KERNEL_SOURCES})
            list(APPEND _kernel_files "${OP_DIR}/${s}")
        endforeach()
        list(APPEND _all_sources ${_kernel_files})
    endif()

    # Create the operator library
    if(_all_sources)
        add_library(op_${OP_NAME} STATIC ${_all_sources})

        # Collect all subdirectories for includes (flat include style)
        set(_op_include_dirs ${CMAKE_SOURCE_DIR}/include ${OP_DIR})
        foreach(_sub IN ITEMS golden host runner kernel bench)
            if(EXISTS "${OP_DIR}/${_sub}")
                list(APPEND _op_include_dirs "${OP_DIR}/${_sub}")
            endif()
        endforeach()

        target_include_directories(op_${OP_NAME} PUBLIC ${_op_include_dirs})
        target_link_libraries(op_${OP_NAME} PUBLIC harness_core)

        if(HARNESS_ENABLE_ASCEND AND OP_HAS_KERNEL)
            target_compile_definitions(op_${OP_NAME} PRIVATE HARNESS_HAS_ASCEND=1)
        endif()
    else()
        message(WARNING "Operator ${OP_NAME} has no source files configured")
    endif()

    # Benchmark target
    if(HARNESS_ENABLE_BENCHMARKS AND OP_HAS_BENCHMARK AND OP_BENCH_SOURCES)
        set(_bench_files "")
        foreach(s ${OP_BENCH_SOURCES})
            list(APPEND _bench_files "${OP_DIR}/${s}")
        endforeach()
        add_executable(bench_${OP_NAME} ${_bench_files})
        target_include_directories(bench_${OP_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${OP_DIR}
        )
        target_link_libraries(bench_${OP_NAME} PRIVATE op_${OP_NAME} harness_core)
        set_target_properties(bench_${OP_NAME} PROPERTIES
            CTest_LABELS "benchmark"
        )
    endif()
endfunction()
