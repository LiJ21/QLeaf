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

find_package(PkgConfig REQUIRED)
pkg_check_modules(HDRHISTOGRAM REQUIRED IMPORTED_TARGET hdr_histogram)

add_library(qleaf INTERFACE)
target_include_directories(qleaf INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(qleaf INTERFACE nlohmann_json::nlohmann_json)

set_source_files_properties(bench/worker.cc PROPERTIES LANGUAGE CUDA)
add_executable(worker_bench bench/worker.cc)
target_link_libraries(worker_bench PRIVATE qleaf benchmark::benchmark)

set_source_files_properties(bench/qleaf_latency_gpu.cu PROPERTIES LANGUAGE CUDA)
add_executable(qleaf_latency bench/qleaf_latency.cc bench/qleaf_latency_gpu.cu)
target_link_libraries(qleaf_latency PRIVATE qleaf PkgConfig::HDRHISTOGRAM)

add_executable(external_latency_cpp bench/external_latency_cpp.cc)
target_link_libraries(external_latency_cpp PRIVATE nlohmann_json::nlohmann_json PkgConfig::HDRHISTOGRAM dl)

option(QLEAF_ENABLE_NVFOREST "Enable nvForest backend in external_latency_cpp when available" ON)
if(QLEAF_ENABLE_NVFOREST)
    set(QLEAF_PY_SITE "${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages" CACHE PATH
        "Python site-packages prefix used for optional RAPIDS/nvForest wheels")
    find_package(CUDAToolkit QUIET)
    find_path(NVFOREST_INCLUDE_DIR nvforest/forest_model.hpp
        HINTS "${QLEAF_PY_SITE}/libnvforest/include")
    find_path(TREELITE_INCLUDE_DIR treelite/c_api.h
        HINTS
          "${QLEAF_PY_SITE}/libtreelite/include"
          "${QLEAF_PY_SITE}/treelite/include"
          "${QLEAF_PY_SITE}/include")
    if(TREELITE_INCLUDE_DIR AND NOT EXISTS "${TREELITE_INCLUDE_DIR}/treelite/tree.h")
        set(TREELITE_INCLUDE_DIR "TREELITE_INCLUDE_DIR-NOTFOUND")
    endif()
    find_path(RAFT_INCLUDE_DIR raft/core/handle.hpp
        HINTS "${QLEAF_PY_SITE}/libraft/include")
    find_path(RMM_INCLUDE_DIR rmm/device_buffer.hpp
        HINTS "${QLEAF_PY_SITE}/librmm/include")
    find_path(RAPIDS_LOGGER_INCLUDE_DIR rapids_logger/logger.hpp
        HINTS "${QLEAF_PY_SITE}/rapids_logger/include")
    find_path(CUDA_CCCL_INCLUDE_DIR cuda/version
        HINTS "${QLEAF_PY_SITE}/nvidia/cuda_cccl/include")
    find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
        HINTS "${QLEAF_PY_SITE}/nvidia/cuda_runtime/include")
    find_library(NVFOREST_LIBRARY nvforest++
        HINTS "${QLEAF_PY_SITE}/libnvforest/lib64")
    find_library(TREELITE_LIBRARY treelite
        HINTS "${QLEAF_PY_SITE}/treelite/lib")
    find_library(RMM_LIBRARY rmm
        HINTS "${QLEAF_PY_SITE}/librmm/lib64")
    find_library(RAPIDS_LOGGER_LIBRARY rapids_logger
        HINTS "${QLEAF_PY_SITE}/rapids_logger/lib64")

    set(QLEAF_NVFOREST_COMPILER_OK ON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
        set(QLEAF_NVFOREST_COMPILER_OK OFF)
    endif()

    if(QLEAF_NVFOREST_COMPILER_OK
       AND CUDAToolkit_FOUND AND NVFOREST_INCLUDE_DIR AND TREELITE_INCLUDE_DIR
       AND RAFT_INCLUDE_DIR AND RMM_INCLUDE_DIR AND RAPIDS_LOGGER_INCLUDE_DIR
       AND CUDA_CCCL_INCLUDE_DIR AND CUDA_RUNTIME_INCLUDE_DIR
       AND NVFOREST_LIBRARY AND TREELITE_LIBRARY AND RMM_LIBRARY
       AND RAPIDS_LOGGER_LIBRARY)
        target_compile_definitions(external_latency_cpp PRIVATE
            QLEAF_HAVE_NVFOREST=1 NVFOREST_ENABLE_GPU)
        target_include_directories(external_latency_cpp PRIVATE
            "${NVFOREST_INCLUDE_DIR}"
            "${TREELITE_INCLUDE_DIR}"
            "${RAFT_INCLUDE_DIR}"
            "${RMM_INCLUDE_DIR}"
            "${RMM_INCLUDE_DIR}/rapids"
            "${RAPIDS_LOGGER_INCLUDE_DIR}"
            "${CUDA_CCCL_INCLUDE_DIR}"
            "${CUDA_RUNTIME_INCLUDE_DIR}")
        target_link_libraries(external_latency_cpp PRIVATE
            "${NVFOREST_LIBRARY}"
            "${TREELITE_LIBRARY}"
            "${RMM_LIBRARY}"
            "${RAPIDS_LOGGER_LIBRARY}"
            CUDA::cudart)
        get_filename_component(NVFOREST_LIBRARY_DIR "${NVFOREST_LIBRARY}" DIRECTORY)
        get_filename_component(TREELITE_LIBRARY_DIR "${TREELITE_LIBRARY}" DIRECTORY)
        get_filename_component(RMM_LIBRARY_DIR "${RMM_LIBRARY}" DIRECTORY)
        get_filename_component(RAPIDS_LOGGER_LIBRARY_DIR "${RAPIDS_LOGGER_LIBRARY}" DIRECTORY)
        set_property(TARGET external_latency_cpp APPEND PROPERTY BUILD_RPATH
            "${NVFOREST_LIBRARY_DIR}"
            "${TREELITE_LIBRARY_DIR}"
            "${RMM_LIBRARY_DIR}"
            "${RAPIDS_LOGGER_LIBRARY_DIR}")
        message(STATUS "external_latency_cpp nvForest backend: enabled")
    else()
        message(STATUS "external_latency_cpp nvForest backend: disabled (requires GCC >= 14 plus CUDA, Treelite headers, and RAPIDS nvForest libs)")
    endif()
endif()

set(QLEAF_COMPILED_BT 256)
if(CMAKE_CUDA_FLAGS MATCHES "QLEAF_BT=([0-9]+)")
    set(QLEAF_COMPILED_BT "${CMAKE_MATCH_1}")
endif()
target_compile_definitions(qleaf_latency PRIVATE QLEAF_BT=${QLEAF_COMPILED_BT})

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
