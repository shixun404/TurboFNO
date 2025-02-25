
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "fused_fft_cgemm.cuh"
#include "fft_radix_2_logN_7_upload_0.cuh"
#include "fft_radix_2_logN_8_upload_0.cuh"
#include "fft_radix_2_logN_9_upload_0.cuh"
#include "fft_radix_2_logN_10_upload_0.cuh"
#include <cufftXt.h>

using DataT = float2;
int thread_bs[4] = {8, 16, 8, 16};
void (*turboFFTArr [4])(float2 *, float2 *, int) ={fft_7, fft_8, fft_9, fft_10};

int main(int argc, char** argv){
    if(argc < 7){
      printf("Usage: %s 0 M N K ntest\n", argv[0]);
      printf("Usage: %s 1 N BS threadblock_bs ntest\n", argv[0]);
      return 1;
    }
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, *FFT_input, *dFFT_input, *FFT_output, *dFFT_output;
    int bs, dimX, dimY, M, N, K, FFT_len, FFT_bs, FFT_input_size;
      bs = atoi(argv[1]);
      dimX = atoi(argv[2]);
      dimY = atoi(argv[3]);
      M = bs * dimX * THREADBLOCK_M;
      FFT_len = dimY;
      FFT_bs = bs * dimX * K;
      FFT_input_size = bs * dimX * dimY * K;
      N = atoi(argv[4]);
      K = atoi(argv[5]);
      int num_tests = atoi(argv[6]) : 1;

      long long int A_size = M * K;
      long long int B_size = N * K;
      long long int C_size = M * N;
      FFT_input = (DataT*)malloc(sizeof(DataT) * FFT_input_size);
      B = (DataT*)malloc(sizeof(DataT) * B_size);
      C = (DataT*)malloc(sizeof(DataT) * C_size);
      C_ref = (DataT*)malloc(sizeof(DataT) * C_size);

      cudaMalloc((void**)&dFFT_input, sizeof(DataT) * FFT_input_size);
      cudaMalloc((void**)&dFFT_output, sizeof(DataT) * FFT_input_size);
      cudaMalloc((void**)&dA, sizeof(DataT) * A_size);
      cudaMalloc((void**)&dB, sizeof(DataT) * B_size);
      cudaMalloc((void**)&dC, sizeof(DataT) * C_size);
      cudaMalloc((void**)&dC_ref, sizeof(DataT) * C_size);

      generate_random_vector((float*)dFFT_input, FFT_input_size * 2);
      generate_random_vector((float*)B, B_size * 2);
      fill_vector((float*)C, 0, C_size * 2);

      cudaMemcpy(dFFT_input, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice);
      cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice);
      cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);
      cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);

      DataT alpha = {1.0, -1.0} , beta = {-1.0, 1.0}; 
      // DataT alpha = {1.0, 0} , beta = {1.0, 0}; 

      
      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N+ dimY) * THREADBLOCK_K ;  

      

      cublasHandle_t handle;   
      cublasCreate(&handle);

      cufftHandle plan;
      cufftCreate(&plan);
  
      cufftPlan1d(&plan, FFT_len, CUFFT_C2C, FFT_bs);
  
  
      cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
                          reinterpret_cast<cufftComplex*>(dFFT_output), 
                          CUFFT_FORWARD);
      cudaDeviceSynchronize();
      direct_copy(dFFT_output, dA);
      cudaDeviceSynchronize();
      cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);     
      cudaDeviceSynchronize();



      fused_fft_cgemm<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_input, dB, dC, alpha, beta);
      cudaDeviceSynchronize();
        
    

      cudaMemcpy(C, dC, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);

      verify_vector((float*)C_ref, (float*)C, M * N * 2);

    return 0;
}