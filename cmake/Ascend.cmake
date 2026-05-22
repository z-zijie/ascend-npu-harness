# Ascend.cmake — Detect and configure Ascend CANN environment

function(harness_find_ascend)
    # Try to locate CANN installation
    set(ASCEND_CANDIDATE_PATHS
        "$ENV{ASCEND_HOME_PATH}"
        "$ENV{ASCEND_TOOLKIT_HOME}"
        "/usr/local/Ascend"
        "/home/developer/Ascend/cann-9.0.0"
    )

    foreach(cand ${ASCEND_CANDIDATE_PATHS})
        if(EXISTS "${cand}")
            set(ASCEND_HOME "${cand}" CACHE PATH "Ascend CANN installation root")
            break()
        endif()
    endforeach()

    if(NOT ASCEND_HOME)
        set(ASCEND_FOUND OFF PARENT_SCOPE)
        message(STATUS "Ascend CANN not found")
        return()
    endif()

    # Find include directories
    set(_inc_dirs
        "${ASCEND_HOME}/include"
        "${ASCEND_HOME}/compiler/include"
        "${ASCEND_HOME}/toolkit/include"
        "${ASCEND_HOME}/opp/op_proto/built-in/inc"
    )
    foreach(d ${_inc_dirs})
        if(EXISTS "${d}")
            list(APPEND ASCEND_INC_DIRS "${d}")
        endif()
    endforeach()

    # Find library directories
    set(_lib_dirs
        "${ASCEND_HOME}/lib64"
        "${ASCEND_HOME}/compiler/lib64"
        "${ASCEND_HOME}/toolkit/lib64"
    )
    foreach(d ${_lib_dirs})
        if(EXISTS "${d}")
            list(APPEND ASCEND_LIB_DIRS "${d}")
        endif()
    endforeach()

    # Find required libraries
    find_library(ASCEND_RUNTIME_LIB ascendcl NAMES libascendcl.so PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
    find_library(ASCEND_GE_LIB ge NAMES libge_runner.so libge.so PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
    find_library(ASCEND_ASCENDCL_LIB ascendcl NAMES libascendcl.so PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)
    find_library(ASCEND_RUNTIME_STUB runtime NAMES libruntime.so PATHS ${ASCEND_LIB_DIRS} NO_DEFAULT_PATH)

    if(ASCEND_ASCENDCL_LIB)
        set(ASCEND_LIBRARIES ${ASCEND_ASCENDCL_LIB} ${ASCEND_GE_LIB} ${ASCEND_RUNTIME_STUB} pthread dl)
        set(ASCEND_FOUND ON)
    else()
        set(ASCEND_FOUND OFF)
    endif()

    set(ASCEND_HOME "${ASCEND_HOME}" PARENT_SCOPE)
    set(ASCEND_FOUND "${ASCEND_FOUND}" PARENT_SCOPE)
    set(ASCEND_INCLUDE_DIRS "${ASCEND_INC_DIRS}" PARENT_SCOPE)
    set(ASCEND_LIBRARY_DIRS "${ASCEND_LIB_DIRS}" PARENT_SCOPE)
    set(ASCEND_LIBRARIES "${ASCEND_LIBRARIES}" PARENT_SCOPE)
endfunction()
