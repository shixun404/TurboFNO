#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>


std::vector<int> parse_line(const std::string& line);

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

// CUDA runtime error checking. Reports the failing call and its source
// location, then continues, so a benchmark sweep still reaches its end.
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                     \
    do {                                                                       \
        cudaError_t turbofno_rt_status = (call);                               \
        fflush(stdout);                                                        \
        if (turbofno_rt_status != cudaSuccess) {                               \
            fprintf(stderr, "%s:%d: %s -> %s (%d)\n", __FILE__, __LINE__,      \
                    #call, cudaGetErrorString(turbofno_rt_status),             \
                    static_cast<int>(turbofno_rt_status));                     \
        }                                                                      \
    } while (0)
#endif  // CUDA_RT_CALL

// FFT error checking. Reports the failing call, then leaves the enclosing
// function with a non-zero status, so only use it where returning is valid.
#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                       \
    do {                                                                       \
        cufftResult turbofno_fft_status = (call);                              \
        fflush(stdout);                                                        \
        if (turbofno_fft_status != CUFFT_SUCCESS) {                            \
            fprintf(stderr, "%s:%d: %s -> FFT status %d\n", __FILE__,          \
                    __LINE__, #call,                                           \
                    static_cast<int>(turbofno_fft_status));                    \
            fflush(stderr);                                                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)
#endif  // CUFFT_CALL

// cublas API error chekcing
#ifndef CUBLAS_CALL
#define CUBLAS_CALL(call)                                                                                      \
    {                                                                                                          \
        cublasStatus_t status = call;                                                                          \
        fflush(stdout); \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                 \
            fprintf(stderr, "%s:%d: %s -> cuBLAS status %d\n", __FILE__,                                       \
                    __LINE__, #call, static_cast<int>(status));                                                \
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


#define MY_MUL(a, b, c) c.x = a.x * b.x - a.y * b.y; c.y = a.y * b.x + a.x * b.y;
#define MY_MUL_REPLACE(a, b, c, d) d.x = a.x * b.x - a.y * b.y; d.y = a.y * b.x + a.x * b.y; c = d;
#define MY_ANGLE2COMPLEX(angle, a) a.x = __cosf(angle); a.y =  __sinf(angle); 


#define turboFFT_ZADD(c, a, b) c.x = a.x + b.x; c.y = a.y + b.y;
#define turboFFT_ZSUB(c, a, b) c.x = a.x - b.x; c.y = a.y - b.y;
#define turboFFT_ZMUL(c, a, b) c.x = a.x * b.x; c.x -= a.y * b.y; c.y = a.y * b.x; c.y += a.x * b.y;
