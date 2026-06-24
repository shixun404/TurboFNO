# Shared build configuration for every TurboFNO fusion variant.
#
# Each variant's CMakeLists.txt sets PROJECT_NAME, the source list (SRC), and
# the include directories, then calls turbofno_configure_target(). This factors
# out the CUDA-vs-ROCm/HIP toolchain selection so the per-variant files stay
# small and identical in intent.
#
# Toolchain selection:
#   -DUSE_HIP=ON  -> compile the .cu sources as HIP, link hipFFT/hipBLAS.
#   default       -> compile as CUDA (nvcc), link cuFFT/cuBLAS (NVIDIA path).
#
# On the HIP path CMAKE_HIP_ARCHITECTURES is left to the user/CMake default so
# any AMD arch can be built with -DCMAKE_HIP_ARCHITECTURES=gfx1100 etc.; nothing
# here hardcodes a wavefront width or a specific gfx target.

function(turbofno_configure_target tgt)
    if(USE_HIP)
        set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
        add_executable(${tgt} ${ARGN})
        target_compile_definitions(${tgt} PRIVATE USE_HIP)
        # Shadow the CUDA system headers the sources include (cuda_runtime.h,
        # cublas_v2.h, cufftXt.h, mma.h, ...) with the hip_compat shims so the
        # CUDA-spelled sources compile unchanged on ROCm. HIP build only.
        target_include_directories(${tgt} BEFORE PRIVATE
            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../hip_compat)
        target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=on>
            $<$<COMPILE_LANGUAGE:CXX>:
              -Wno-unused-function
              -Wno-unused-variable
              -Wno-unused-parameter
              -Wno-unused-but-set-variable
              -Wno-unused-result
            >
        )
        find_package(hipfft REQUIRED)
        find_package(hipblas REQUIRED)
        target_link_libraries(${tgt} PRIVATE hip::hipfft roc::hipblas)
    else()
        add_executable(${tgt} ${ARGN})
        target_compile_options(${tgt}
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
        target_link_libraries(${tgt} PRIVATE CUDA::cublas CUDA::cufft)
    endif()
endfunction()
