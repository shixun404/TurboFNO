
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "TurboFNO.h"
#include "cgemm.cuh"
#include <vector>

using DataT = float2;

int main(int argc, char** argv){
    DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, 
          *FFT_input, *dFFT_input, *dFFT_output, 
          *iFFT_output, *diFFT_output, *iFFT_output_ref, *diFFT_output_ref;
    long long int bs, dimX, dimY, DY, M, N, K, FFT_len, FFT_bs, iFFT_bs, FFT_input_size, iFFT_output_size;
    double epsilon = 1e-6;
    // 默认值
    bs = 128;
    dimX = 256;
    DY = 256;
    N = 128;
    K = 128;

    // 解析命令行参数
    if (argc > 1) {
        if (argc != 7) {
            printf("Usage: %s <bs> <dimX> <DY> <N> <K>\n", argv[0]);
            printf("Example: %s 128 256 256 128 128\n", argv[0]);
            printf("Using default values: bs=%lld, dimX=%lld, DY=%lld, N=%lld, K=%lld\n", 
                   bs, dimX, DY, N, K);
        } else {
            bs = atoi(argv[1]);
            dimX = atoi(argv[2]);
            DY = atoi(argv[3]);
            N = atoi(argv[4]);
            K = atoi(argv[5]);
            epsilon = atof(argv[6]);
        }
    }

    printf("Parameters: bs=%lld, dimX=%lld, DY=%lld, N=%lld, K=%lld\n", 
           bs, dimX, DY, N, K);

    ntest = 5;

    M = bs * dimX * THREADBLOCK_M;
    dimY = 64;
    FFT_len = DY;
    FFT_bs = bs * dimX * K;
    iFFT_bs = bs * dimX * N;
    FFT_input_size = bs * dimX * DY * K;
    iFFT_output_size = bs * dimX * DY * N;

      long long int A_size = M * K;
      long long int B_size = N * K;
      long long int C_size = M * N;
      
      A = (DataT*)malloc(sizeof(DataT) * (A_size + ntest));
      B = (DataT*)malloc(sizeof(DataT) * (B_size + ntest));
      C = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));
      C_ref = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));


      
      CUDA_RT_CALL(cudaMalloc((void**)&dA, sizeof(DataT) * (A_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dB, sizeof(DataT) * (B_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC, sizeof(DataT) * (C_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC_ref, sizeof(DataT) * (C_size + ntest)));

      generate_random_vector((float*)A, A_size * 2);
      generate_random_vector((float*)B, B_size * 2);
      fill_vector((float*)C, 0, C_size * 2);


      CUDA_RT_CALL(cudaMemcpy(dA, A, sizeof(DataT) * A_size, cudaMemcpyHostToDevice));
      
      CUDA_RT_CALL(cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));

      DataT alpha = {1.0, -1.0} , beta = {-1.0, 1.0}; 
      

      M = bs * dimX * THREADBLOCK_M;
      dimY = 64;
      FFT_len = DY;
      FFT_bs = bs * dimX * K;
      iFFT_bs = bs * dimX * N;
      FFT_input_size = bs * dimX * DY * K;
      iFFT_output_size = bs * dimX * DY * N;

      A_size = M * K;
      B_size = N * K;
      C_size = M * N;
      
      cublasHandle_t handle;   
      cublasCreate(&handle);

                
      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K * 2;

      cudaDeviceSynchronize();
            
      CUBLAS_CALL(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M));
      cgemm<<<gridDim, blockDim, shmem_size>>>(M, N, K, dA, dB, dC, alpha, beta);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      cudaDeviceSynchronize();
      
      cudaMemcpy(C_ref, dC_ref, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C, dC, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
    
      verify_vector((float*)C_ref, (float*)C, C_size * 2);

      // {
      //   cudaEvent_t fft_begin, fft_end;
      //   float elapsed_time;
      //   cudaEventCreate(&fft_begin);
      //   cudaEventCreate(&fft_end);
      // cudaEventRecord(fft_begin);
      // for (int i = 0; i < ntest; ++i){
      //   cgemm<<<gridDim, blockDim, shmem_size>>>(M, N, K, dA, dB, dC, alpha, beta);
      //   cudaDeviceSynchronize();
      // }
      // cudaEventRecord(fft_end);
      // cudaEventSynchronize(fft_begin);
      // cudaEventSynchronize(fft_end);
      // cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      // elapsed_time = elapsed_time / ntest;
      // // printf("bs=%d dimX=%d DY=%d M=%d, N=%d K=%d\n", bs, dimX, DY, M, N, K);
      // // printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      // printf("1D_A, bs=%-4d, dimX=%-4d, DY=%-4d, N=%-4d, K=%-4d, TIME=%8.3fms\n",
      //   bs, dimX, DY, N, K, elapsed_time);
    // }

    return 0;
}