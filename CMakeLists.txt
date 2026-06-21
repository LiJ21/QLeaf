cmake_minimum_required(VERSION 3.24)

option(QLEAF_BUILD_TESTS "Build tests" ON)
option(QLEAF_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(QLEAF_BUILD_CUDA "Build CUDA targets" OFF)
option(QLEAF_ENABLE_COMPILER_CACHE
    "Use sccache/ccache as compiler launcher when available" ON)

if(QLEAF_BUILD_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # RTX 3060 is sm_86. The -real suffix asks CMake to emit only SASS for
    # this GPU instead of also embedding PTX; users can still override this at
    # configure time with -DCMAKE_CUDA_ARCHITECTURES=...
    set(CMAKE_CUDA_ARCHITECTURES "86-real" CACHE STRING
        "CUDA architectures to build; qleaf latency benchmarks default to RTX 3060")
endif()

project(qleaf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(QLEAF_BUILD_CUDA)
    enable_language(CUDA)
    # nvcc 12.0 caps device code at C++20 (host stays C++23 above).
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(QLEAF_ENABLE_COMPILER_CACHE)
    find_program(QLEAF_COMPILER_CACHE sccache ccache)
    if(QLEAF_COMPILER_CACHE)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${QLEAF_COMPILER_CACHE}")
        if(QLEAF_BUILD_CUDA)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${QLEAF_COMPILER_CACHE}")
        endif()
        message(STATUS "Compiler cache launcher: ${QLEAF_COMPILER_CACHE}")
    endif()
endif()

add_library(qleaf INTERFACE)
target_include_directories(qleaf INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

include(FetchContent)

if(QLEAF_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)

    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.fc_deps/googletest-src/CMakeLists.txt")
            FetchContent_Declare(googletest
                SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.fc_deps/googletest-src")
        else()
            FetchContent_Declare(googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG        v1.14.0)
        endif()
        FetchContent_MakeAvailable(googletest)
    endif()

    file(GLOB_RECURSE qleaf_TEST_SOURCES CONFIGURE_DEPENDS test/*.cc)

    foreach(test_src IN LISTS qleaf_TEST_SOURCES)
        file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR}/test ${test_src})
        string(REPLACE "/" "_" target_name ${rel_path})
        string(REPLACE "\\" "_" target_name ${target_name})
        string(REPLACE ".cc" "" target_name ${target_name})
        set(target_name test_${target_name})

        add_executable(${target_name} ${test_src})
        target_link_libraries(${target_name} PRIVATE qleaf GTest::gtest_main)
        gtest_discover_tests(${target_name})
    endforeach()
endif()

if(QLEAF_BUILD_BENCHMARKS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(HDRHISTOGRAM REQUIRED IMPORTED_TARGET hdr_histogram)

    add_executable(qleaf_latency bench/qleaf_latency.cc)
    target_link_libraries(qleaf_latency PRIVATE qleaf PkgConfig::HDRHISTOGRAM)

    set(QLEAF_LATENCY_GPU_DEFS)
    if(QLEAF_BUILD_CUDA)
        set(QLEAF_LATENCY_GPU_SUITES "best" CACHE STRING
            "GPU suites compiled into qleaf_latency: best, all, sp, sp-map, mp-sentinel, bitmask-tpt, bitmask-cache, oneshot, naive, ablation")
        set(QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu.cu)
        set(QLEAF_LATENCY_GPU_SELECTED)
        foreach(suite IN LISTS QLEAF_LATENCY_GPU_SUITES)
            if(suite STREQUAL "all")
                list(APPEND QLEAF_LATENCY_GPU_SELECTED
                    sp sp-map mp-sentinel bitmask-tpt bitmask-cache oneshot naive ablation)
            elseif(suite STREQUAL "best")
                list(APPEND QLEAF_LATENCY_GPU_SELECTED sp bitmask-tpt)
            elseif(suite STREQUAL "sp" OR suite STREQUAL "sp-map" OR
                   suite STREQUAL "mp-sentinel" OR suite STREQUAL "bitmask-tpt" OR
                   suite STREQUAL "bitmask-cache" OR suite STREQUAL "oneshot" OR
                   suite STREQUAL "naive" OR suite STREQUAL "ablation")
                list(APPEND QLEAF_LATENCY_GPU_SELECTED "${suite}")
            else()
                message(FATAL_ERROR "Unknown QLEAF_LATENCY_GPU_SUITES entry: ${suite}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES QLEAF_LATENCY_GPU_SELECTED)
        foreach(suite IN LISTS QLEAF_LATENCY_GPU_SELECTED)
            if(suite STREQUAL "sp")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_sp.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_SP=1)
            elseif(suite STREQUAL "sp-map")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_sp_map.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_SP_MAP=1)
            elseif(suite STREQUAL "mp-sentinel")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_mp_sentinel.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_MP_SENTINEL=1)
            elseif(suite STREQUAL "bitmask-tpt")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_bitmask_tpt.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_BITMASK_TPT=1)
            elseif(suite STREQUAL "bitmask-cache")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_bitmask_cache.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_BITMASK_CACHE=1)
            elseif(suite STREQUAL "oneshot")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_oneshot.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_ONESHOT=1)
            elseif(suite STREQUAL "naive")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_naive.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_NAIVE=1)
            elseif(suite STREQUAL "ablation")
                list(APPEND QLEAF_LATENCY_GPU_SOURCES bench/qleaf_latency_gpu_ablation.cu)
                list(APPEND QLEAF_LATENCY_GPU_DEFS QLEAF_GPU_HAS_ABLATION=1)
            endif()
        endforeach()
        message(STATUS "qleaf_latency GPU suites: ${QLEAF_LATENCY_GPU_SELECTED}")
        target_sources(qleaf_latency PRIVATE ${QLEAF_LATENCY_GPU_SOURCES})
        target_compile_definitions(qleaf_latency PRIVATE ${QLEAF_LATENCY_GPU_DEFS})
    else()
        target_sources(qleaf_latency PRIVATE bench/qleaf_latency_gpu_stub.cc)
    endif()

    set(QLEAF_COMPILED_BT 256)
    if(QLEAF_BUILD_CUDA AND CMAKE_CUDA_FLAGS MATCHES "QLEAF_BT=([0-9]+)")
        set(QLEAF_COMPILED_BT "${CMAKE_MATCH_1}")
    endif()
    target_compile_definitions(qleaf_latency PRIVATE QLEAF_BT=${QLEAF_COMPILED_BT})

    add_executable(external_latency_cpp bench/external_latency_cpp.cc)
    target_link_libraries(external_latency_cpp PRIVATE qleaf PkgConfig::HDRHISTOGRAM dl)

    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(external_latency_cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()

    option(QLEAF_ENABLE_NVFOREST
        "Enable nvForest backend in external_latency_cpp when available" ON)
    if(QLEAF_ENABLE_NVFOREST)
        set(QLEAF_PY_SITE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.12/site-packages" CACHE PATH
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

    if(QLEAF_BUILD_CUDA)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.fc_deps/googlebenchmark-src/CMakeLists.txt")
            FetchContent_Declare(googlebenchmark
                SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.fc_deps/googlebenchmark-src")
        else()
            FetchContent_Declare(googlebenchmark
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG        v1.8.3)
        endif()
        FetchContent_MakeAvailable(googlebenchmark)

        set_source_files_properties(bench/worker.cc PROPERTIES LANGUAGE CUDA)
        add_executable(worker_bench bench/worker.cc)
        set(QLEAF_WORKER_CUDA_SUITES "best" CACHE STRING
            "CUDA suites compiled into worker_bench: best, all, oneshot-combos, persistent-combos, sp, sp-cache, bitmask-tpt, bitmask-oneshot")
        set(QLEAF_WORKER_CUDA_ONESHOT_COMBOS 0)
        set(QLEAF_WORKER_CUDA_PERSISTENT_COMBOS 0)
        set(QLEAF_WORKER_CUDA_SP 0)
        set(QLEAF_WORKER_CUDA_SP_CACHE 0)
        set(QLEAF_WORKER_CUDA_BITMASK_TPT 0)
        set(QLEAF_WORKER_CUDA_BITMASK_ONESHOT 0)
        set(QLEAF_WORKER_CUDA_SELECTED)
        foreach(suite IN LISTS QLEAF_WORKER_CUDA_SUITES)
            if(suite STREQUAL "all")
                list(APPEND QLEAF_WORKER_CUDA_SELECTED
                    oneshot-combos persistent-combos sp sp-cache bitmask-tpt bitmask-oneshot)
            elseif(suite STREQUAL "best")
                list(APPEND QLEAF_WORKER_CUDA_SELECTED sp bitmask-tpt)
            elseif(suite STREQUAL "oneshot-combos" OR suite STREQUAL "persistent-combos" OR
                   suite STREQUAL "sp" OR suite STREQUAL "sp-cache" OR
                   suite STREQUAL "bitmask-tpt" OR suite STREQUAL "bitmask-oneshot")
                list(APPEND QLEAF_WORKER_CUDA_SELECTED "${suite}")
            else()
                message(FATAL_ERROR "Unknown QLEAF_WORKER_CUDA_SUITES entry: ${suite}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES QLEAF_WORKER_CUDA_SELECTED)
        foreach(suite IN LISTS QLEAF_WORKER_CUDA_SELECTED)
            if(suite STREQUAL "oneshot-combos")
                set(QLEAF_WORKER_CUDA_ONESHOT_COMBOS 1)
            elseif(suite STREQUAL "persistent-combos")
                set(QLEAF_WORKER_CUDA_PERSISTENT_COMBOS 1)
            elseif(suite STREQUAL "sp")
                set(QLEAF_WORKER_CUDA_SP 1)
            elseif(suite STREQUAL "sp-cache")
                set(QLEAF_WORKER_CUDA_SP_CACHE 1)
            elseif(suite STREQUAL "bitmask-tpt")
                set(QLEAF_WORKER_CUDA_BITMASK_TPT 1)
            elseif(suite STREQUAL "bitmask-oneshot")
                set(QLEAF_WORKER_CUDA_BITMASK_ONESHOT 1)
            endif()
        endforeach()
        message(STATUS "worker_bench CUDA suites: ${QLEAF_WORKER_CUDA_SELECTED}")
        target_compile_definitions(worker_bench PRIVATE
            QLEAF_WORKER_CUDA_ONESHOT_COMBOS=${QLEAF_WORKER_CUDA_ONESHOT_COMBOS}
            QLEAF_WORKER_CUDA_PERSISTENT_COMBOS=${QLEAF_WORKER_CUDA_PERSISTENT_COMBOS}
            QLEAF_WORKER_CUDA_SP=${QLEAF_WORKER_CUDA_SP}
            QLEAF_WORKER_CUDA_SP_CACHE=${QLEAF_WORKER_CUDA_SP_CACHE}
            QLEAF_WORKER_CUDA_BITMASK_TPT=${QLEAF_WORKER_CUDA_BITMASK_TPT}
            QLEAF_WORKER_CUDA_BITMASK_ONESHOT=${QLEAF_WORKER_CUDA_BITMASK_ONESHOT})
        target_link_libraries(worker_bench PRIVATE qleaf benchmark::benchmark)
    endif()
endif()
