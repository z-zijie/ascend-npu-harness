# GTest.cmake — Find and configure Google Test

# Try to find system GTest first
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "Found system GTest ${GTest_VERSION}")
    # GTest::GTest and GTest::Main are available
    set(HARNESS_GTEST_TARGET "GTest::GTest")
    set(HARNESS_GTEST_MAIN_TARGET "GTest::Main")
    set(HARNESS_GTEST_FOUND ON)
else()
    # Try to find gtest from known system paths
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h
        PATHS /usr/include /usr/local/include
    )
    find_library(GTEST_LIB gtest
        PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
    )
    find_library(GTEST_MAIN_LIB gtest_main
        PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
    )

    if(GTEST_INCLUDE_DIR AND GTEST_LIB AND GTEST_MAIN_LIB)
        message(STATUS "Found GTest manually at ${GTEST_INCLUDE_DIR}")
        add_library(GTest::GTest UNKNOWN IMPORTED)
        set_target_properties(GTest::GTest PROPERTIES
            IMPORTED_LOCATION "${GTEST_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
        )
        add_library(GTest::Main UNKNOWN IMPORTED)
        set_target_properties(GTest::Main PROPERTIES
            IMPORTED_LOCATION "${GTEST_MAIN_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
        )
        set(HARNESS_GTEST_TARGET "GTest::GTest" PARENT_SCOPE)
        set(HARNESS_GTEST_MAIN_TARGET "GTest::Main" PARENT_SCOPE)
        set(HARNESS_GTEST_FOUND ON PARENT_SCOPE)
    else()
        message(WARNING "GTest not found; tests disabled")
        set(HARNESS_GTEST_FOUND OFF PARENT_SCOPE)
    endif()
endif()
