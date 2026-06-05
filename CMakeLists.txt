cmake_minimum_required(VERSION 3.24)
project(qleaf_tests CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nvcc 12.0 caps device code at C++20 (host stays C++23 above).
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# cuda::atomic requires sm_60+ on Linux; users can still override this at
# configure time with -DCMAKE_CUDA_ARCHITECTURES=...
set(CMAKE_CUDA_ARCHITECTURES 60)

include(FetchContent)

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3)

FetchContent_MakeAvailable(googletest nlohmann_json googlebenchmark)

add_library(qleaf INTERFACE)
target_include_directories(qleaf INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qleaf INTERFACE nlohmann_json::nlohmann_json)

set_source_files_properties(bench/worker.cc PROPERTIES LANGUAGE CUDA)
add_executable(worker_bench bench/worker.cc)
target_link_libraries(worker_bench PRIVATE qleaf benchmark::benchmark)

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
