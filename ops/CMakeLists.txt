# ops/CMakeLists.txt — Operator subdirectory management
# Each operator directory has its own CMakeLists.txt

# Operator directories (in ladder order)
set(OPERATOR_DIRS
    elementwise_add
    axpy_scalar_affine
    broadcast_binary
    activation_gelu_silu
    reduce_sum_lastdim
    transpose_2d_4d
    layer_norm
    matmul_bias
    rms_norm_rope_fused
    flash_attention_fwd_fused
)

foreach(op_dir ${OPERATOR_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${op_dir}/CMakeLists.txt")
        add_subdirectory(${op_dir})
    endif()
endforeach()
