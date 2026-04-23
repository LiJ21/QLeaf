cmake_minimum_required(VERSION 3.24)
project(qleaf_tests CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)

FetchContent_MakeAvailable(googletest nlohmann_json)

add_library(qleaf INTERFACE)
target_include_directories(qleaf INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qleaf INTERFACE nlohmann_json::nlohmann_json)

file(GLOB_RECURSE qleaf_TEST_SOURCES CONFIGURE_DEPENDS test/*.cc)

foreach(test_src IN LISTS qleaf_TEST_SOURCES)
    file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR}/test ${test_src})
    string(REPLACE "/" "_" target_name ${rel_path})
    string(REPLACE "\\" "_" target_name ${target_name})
    string(REPLACE ".cc" "" target_name ${target_name})
    set(target_name test_${target_name})

    add_executable(${target_name} ${test_src})
    target_link_libraries(${target_name} PRIVATE qleaf GTest::gtest_main)
endforeach()

include(GoogleTest)