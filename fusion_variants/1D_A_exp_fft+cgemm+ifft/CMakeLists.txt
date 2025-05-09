cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TurboFNO_1D_A LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# === 用户需手动 export 环境变量 PROJECT_ROOT ===
if(NOT DEFINED ENV{PROJECT_ROOT})
    message(FATAL_ERROR "Please set the PROJECT_ROOT environment variable. Example: export PROJECT_ROOT=/path/to/TurboFNO")
endif()

set(PROJECT_ROOT $ENV{PROJECT_ROOT})

add_compile_definitions(DEFAULT_CONFIG_PATH="${PROJECT_ROOT}/benchmark_config/problem_size_1d.txt")

include_directories(
    ${PROJECT_ROOT}/TurboFFT/Common
    ${PROJECT_ROOT}/utils
    ${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2
)

file(GLOB VARIANT_SOURCES
    "${CMAKE_SOURCE_DIR}/*.cu"
    "${PROJECT_ROOT}/utils/*.cu"
)

add_executable(${PROJECT_NAME} ${VARIANT_SOURCES})

target_compile_options(${PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
      -Wno-unused-function
      -Wno-unused-variable
      -Wno-unused-parameter
      -Wno-unused-but-set-variable
      -Wno-unused-result
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
      --disable-warnings
    >
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE CUDA::cublas CUDA::cufft
)