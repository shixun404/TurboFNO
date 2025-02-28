#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define CHECK_CUDA_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    fflush(stdout); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Kernel Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
    cudaDeviceSynchronize(); \
}


#define CUDA_CALLER(call) do{\
  cudaError_t cuda_ret = (call);\
  fflush(stdout); \
  if(cuda_ret != cudaSuccess){\
    printf("CUDA Error at line %d in file %s\n", __LINE__, __FILE__);\
    printf("  Error message: %s\n", cudaGetErrorString(cuda_ret));\
    printf("  In the function call %s\n", #call);\
    exit(1);\
  }\
}while(0)

#pragma once

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        fflush(stdout); \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        fflush(stdout); \
        if ( status != CUFFT_SUCCESS )  {                                                                               \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
                     fflush(stdout);\
                     return 1;} \
    }
#endif  // CUFFT_CALL

// cublas API error chekcing
#ifndef CUBLAS_CALL
#define CUBLAS_CALL(call)                                                                                      \
    {                                                                                                          \
        cublasStatus_t status = call;                                                                          \
        fflush(stdout); \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                 \
            fprintf(stderr,                                                                                    \
                    "ERROR: cuBLAS call \"%s\" failed in line %d of file %s with error code (%d).\n",          \
                    #call, __LINE__, __FILE__, status);                                                        \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    }
#endif

#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)


#define Z_SUB(a, b, c) c.x = a.x - b.x; c.y = a.y - b.y;
#define Z_ADD(a, b, c) c.x = a.x + b.x; c.y = a.y + b.y;
#define Z_MUL(a, b, c) c.x += a.x * b.x - a.y * b.y; c.y += a.y * b.x + a.x * b.y;

class saxpy_timer
{
public:
    saxpy_timer() { reset(); }
    void reset() {
    t0_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed(bool reset_timer=false) {
    std::chrono::high_resolution_clock::time_point t =
            std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t - t0_);
    if (reset_timer)
        reset();
    return time_span.count();
    }
    double elapsed_msec(bool reset_timer=false) {
    return elapsed(reset_timer) * 1000;
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};

//__global__ void fill(float *a , float x, int N);

cudaDeviceProp getDetails(int deviceId);

void generate_random_vector(float* target, int n);

void copy_vector(float *src, float *dest, int n);

bool verify_vector(float *vec1, float *vec2, int n, int nrow);

void fill_vector(float*, int, float);

void copy_matrix(float *src, float *dest, int n);

void copy_matrix_double(double *src, double *dest, int n);

void generate_random_matrix(float* target, int n);

void generate_random_matrix_double(double* target, int n);

bool verify_matrix(float*, float*, int n);

bool verify_matrix_double(double*, double*, int n);

bool verify_matrix_double2(double*, double*, int n);

void cpu_gemm(float alpha, float beta, float *mat1, float*mat2, int max_size, float* mat3);

void print_matrix(float*, int);


/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#pragma once

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL

// template <> struct traits<CUFFT_C2C> {
//     // scalar type
//     typedef float T;

//     using input_host_type = std::complex<T>;
//     using input_device_type = cufftComplex;

//     using output_host_type = std::complex<T>;
//     using output_device_type = cufftComplex;

//     static constexpr cufftType_t transformType = CUDA_R_64F;

//     template <typename RNG> inline static T rand(RNG &gen) {
//         return make_cuFloatComplex((S)gen(), (S)gen());
//     }
// };


#define MY_MUL(a, b, c) c.x = a.x * b.x - a.y * b.y; c.y = a.y * b.x + a.x * b.y;
#define MY_MUL_REPLACE(a, b, c, d) d.x = a.x * b.x - a.y * b.y; d.y = a.y * b.x + a.x * b.y; c = d;
#define MY_ANGLE2COMPLEX(angle, a) a.x = __cosf(angle); a.y =  __sinf(angle); 


#define turboFFT_ZADD(c, a, b) c.x = a.x + b.x; c.y = a.y + b.y;
#define turboFFT_ZSUB(c, a, b) c.x = a.x - b.x; c.y = a.y - b.y;
#define turboFFT_ZMUL(c, a, b) c.x = a.x * b.x; c.x -= a.y * b.y; c.y = a.y * b.x; c.y += a.x * b.y;
