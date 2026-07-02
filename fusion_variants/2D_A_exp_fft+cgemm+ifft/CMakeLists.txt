cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

option(USE_HIP "Build for ROCm/HIP instead of CUDA" OFF)

if(USE_HIP)
    project(TurboFNO_2D_A LANGUAGES CXX HIP)
else()
    project(TurboFNO_2D_A LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
else()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

if(NOT DEFINED ENV{PROJECT_ROOT})
    message(FATAL_ERROR "Please set the PROJECT_ROOT environment variable. Example: export PROJECT_ROOT=/path/to/TurboFNO")
endif()

set(PROJECT_ROOT $ENV{PROJECT_ROOT})
add_compile_definitions(DEFAULT_CONFIG_PATH="${PROJECT_ROOT}/benchmark_config/problem_size_2d.txt")

include(${PROJECT_ROOT}/cmake/turbofno_targets.cmake)

include_directories(
    ${PROJECT_ROOT}/TurboFFT/Common
    ${PROJECT_ROOT}/utils
    ${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2
)

file(GLOB VARIANT_SOURCES
    "${CMAKE_SOURCE_DIR}/*.cu"
    "${PROJECT_ROOT}/utils/*.cu"
)

turbofno_configure_target(${PROJECT_NAME} ${VARIANT_SOURCES})
