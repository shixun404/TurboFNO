cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(FT "0" CACHE STRING "Description of FT")
# set(FT $ENV{FT} CACHE STRING "Compile-time parameter FT")

# Include directories
# include_directories(/global/homes/s/swu264/TurboFNO/cgemm_cutlass/include)
include_directories(/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/include)
include_directories(/global/homes/s/swu264/TurboFNO/cgemm/Common)
include_directories(/global/homes/s/swu264/TurboFNO/utils)
include_directories(${CMAKE_SOURCE_DIR})

# CUDA paths
set(CUDA_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda")
set(MATH_LIB "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4")

# Library directories
link_directories(${MATH_LIB}/lib64)


add_executable(cgemm main.cu /global/homes/s/swu264/TurboFNO/utils/utils.cu cgemm.cuh)
target_compile_options(cgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(cgemm PRIVATE cublas)

# Clean the build
add_custom_target(clean-all
   COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/CleanAll.cmake
)

# Note: You might need to create a CleanAll.cmake file or an alternative way to clean your build files, as CMake's clean command does not remove the executables.