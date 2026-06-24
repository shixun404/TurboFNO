#pragma once

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>
//
// CUDA-to-HIP compatibility shim for the ROCm/HIP build.
//
// This directory (hip_compat) is placed on the include path only for the HIP
// build (see cmake/turbofno_targets.cmake). The thin headers next to this one
// -- cuda_runtime.h, cublas_v2.h, cufftXt.h, cufft.h, mma.h, helper_*.h --
// shadow the CUDA system headers the sources include, so the project's
// CUDA-spelled sources compile unchanged on ROCm. This header includes the HIP
// runtime plus hipFFT/hipBLAS and aliases the small set of cuda*/cufft*/cublas*
// spellings the project uses to their HIP equivalents. The hand-written device
// kernels (the float2 FFT and the SIMT complex GEMM) need no aliasing; they are
// portable HIP/CUDA C++.

#if !defined(USE_HIP) && !defined(__HIP_PLATFORM_AMD__)
#error "hip_compat is for the HIP build only; do not add it to the CUDA include path"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipfft/hipfft.h>
#include <hipblas/hipblas.h>

// Runtime API
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties

// Event timing
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// Complex type used in cuBLAS/cuFFT casts (both are float2 underneath)
#define cuFloatComplex hipFloatComplex
#define make_cuFloatComplex make_hipFloatComplex

// cuFFT -> hipFFT
#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define cufftType hipfftType
#define cufftComplex hipfftComplex
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define cufftCreate hipfftCreate
#define cufftDestroy hipfftDestroy
#define cufftPlan1d hipfftPlan1d
#define cufftPlanMany hipfftPlanMany
#define cufftExecC2C hipfftExecC2C

// cuBLAS -> hipBLAS
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasCgemm hipblasCgemm
