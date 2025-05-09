cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TurboFNO_2D_B LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# =========== Check PROJECT_ROOT env var ===========
if(NOT DEFINED ENV{PROJECT_ROOT})
    message(FATAL_ERROR "❌ Please set the PROJECT_ROOT environment variable. For example:\n  export PROJECT_ROOT=/path/to/TurboFNO")
endif()

set(PROJECT_ROOT $ENV{PROJECT_ROOT})

# =========== Include directories ===========
include_directories(
    ${PROJECT_ROOT}/Common
    ${PROJECT_ROOT}/utils
    ${PROJECT_ROOT}/TurboFFT/TurboFFT/include/code_gen/generated/float2
)

# =========== Source files ===========
set(SRC
    fused_trunc_2D.cu
    ${PROJECT_ROOT}/utils/utils.cu
)

# =========== Build ===========
add_executable(${PROJECT_NAME} ${SRC})

target_compile_options(${PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE CUDA::cublas CUDA::cufft
)

# =========== Optional clean-all ===========
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
)