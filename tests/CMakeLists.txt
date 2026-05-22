# tests/CMakeLists.txt — Test targets

include(../cmake/GTest)
include(GoogleTest)

# Helper: add a host test
function(harness_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ops
    )
    target_link_libraries(${TEST_NAME} PRIVATE harness_core)

    # Link operator library if it matches the test name pattern test_<op>
    string(REGEX REPLACE "^test_" "" _op_name "${TEST_NAME}")
    if(TARGET op_${_op_name})
        target_link_libraries(${TEST_NAME} PRIVATE op_${_op_name})
        # Add op subdirectories for flat includes
        set(_op_dir "${CMAKE_SOURCE_DIR}/ops/${_op_name}")
        foreach(_sub IN ITEMS golden host runner kernel)
            if(EXISTS "${_op_dir}/${_sub}")
                target_include_directories(${TEST_NAME} PRIVATE "${_op_dir}/${_sub}")
            endif()
        endforeach()
    endif()

    # GTest linkage
    if(HARNESS_GTEST_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${HARNESS_GTEST_TARGET} ${HARNESS_GTEST_MAIN_TARGET})
    else()
        target_link_libraries(${TEST_NAME} PRIVATE gtest gtest_main pthread)
    endif()

    # Labels
    set(_labels host golden)
    if(${TEST_NAME} MATCHES "flash_attention")
        list(APPEND _labels flash_attention attention llm)
    elseif(${TEST_NAME} MATCHES "rms_norm_rope")
        list(APPEND _labels llm fusion reduction elementwise)
    elseif(${TEST_NAME} MATCHES "matmul")
        list(APPEND _labels matmul)
    elseif(${TEST_NAME} MATCHES "layer_norm")
        list(APPEND _labels normalization reduction elementwise)
    elseif(${TEST_NAME} MATCHES "transpose")
        list(APPEND _labels layout)
    elseif(${TEST_NAME} MATCHES "reduce")
        list(APPEND _labels reduction)
    elseif(${TEST_NAME} MATCHES "activation")
        list(APPEND _labels elementwise)
    elseif(${TEST_NAME} MATCHES "broadcast")
        list(APPEND _labels elementwise)
    elseif(${TEST_NAME} MATCHES "axpy")
        list(APPEND _labels elementwise)
    elseif(${TEST_NAME} MATCHES "elementwise_add")
        list(APPEND _labels elementwise)
    endif()

    set_target_properties(${TEST_NAME} PROPERTIES LABELS "${_labels}")

    gtest_discover_tests(${TEST_NAME}
        PROPERTIES LABELS "${_labels}"
        DISCOVERY_TIMEOUT 30
    )
endfunction()

# Operator tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_elementwise_add.cpp")
    harness_add_test(test_elementwise_add test_elementwise_add.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_axpy_scalar_affine.cpp")
    harness_add_test(test_axpy_scalar_affine test_axpy_scalar_affine.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_broadcast_binary.cpp")
    harness_add_test(test_broadcast_binary test_broadcast_binary.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_activation_gelu_silu.cpp")
    harness_add_test(test_activation_gelu_silu test_activation_gelu_silu.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_reduce_sum_lastdim.cpp")
    harness_add_test(test_reduce_sum_lastdim test_reduce_sum_lastdim.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_transpose_2d_4d.cpp")
    harness_add_test(test_transpose_2d_4d test_transpose_2d_4d.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_layer_norm.cpp")
    harness_add_test(test_layer_norm test_layer_norm.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_matmul_bias.cpp")
    harness_add_test(test_matmul_bias test_matmul_bias.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_rms_norm_rope_fused.cpp")
    harness_add_test(test_rms_norm_rope_fused test_rms_norm_rope_fused.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_flash_attention_fwd_fused.cpp")
    harness_add_test(test_flash_attention_fwd_fused test_flash_attention_fwd_fused.cpp)
endif()

# Harness core tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_harness_core.cpp")
    harness_add_test(test_harness_core test_harness_core.cpp)
endif()
