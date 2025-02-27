
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "fused_fft_cgemm_ifft_trunc.cuh"
#include <cufftXt.h>

using DataT = float2;
int thread_bs[4] = {8, 16, 8, 16};

__global__ void direct_copy_colmajor_float4_truncation(const float2 *input,
  float2       *output,
  int M, int K, int reduced_M)
{
  const float4 *inputF4  = reinterpret_cast<const float4*>(input);
  float4       *outputF4 = reinterpret_cast<float4*>(output);

  int inputFloat4PerCol  = M / 2;         // total float4 chunks in each column
  int outputFloat4PerCol = reduced_M / 2; // float4 chunks in truncated top rows

  int startRow4 = blockIdx.x * blockDim.x + threadIdx.x; 
  int startCol  = blockIdx.y * blockDim.y + threadIdx.y;

  int strideRow4 = blockDim.x * gridDim.x;
  int strideCol  = blockDim.y * gridDim.y;

  for (int col = startCol; col < K; col += strideCol) {
    for (int row4 = startRow4; row4 < outputFloat4PerCol; row4 += strideRow4) {
    int inIndex  = col * inputFloat4PerCol  + row4;
    int outIndex = col * outputFloat4PerCol + row4;

    outputF4[outIndex] = inputF4[inIndex];
    }
  }
}

__global__ void direct_copy_colmajor_float4_zero_padding(const float2 *input,
  float2       *output,
  int M, int K, int reduced_M)
{
  const float4 *inputF4  = reinterpret_cast<const float4*>(input);
  float4       *outputF4 = reinterpret_cast<float4*>(output);

  int inputFloat4PerCol  = reduced_M / 2;         // total float4 chunks in each column
  int outputFloat4PerCol = M / 2; // float4 chunks in truncated top rows

  int startRow4 = blockIdx.x * blockDim.x + threadIdx.x; 
  int startCol  = blockIdx.y * blockDim.y + threadIdx.y;

  int strideRow4 = blockDim.x * gridDim.x;
  int strideCol  = blockDim.y * gridDim.y;
  float4 zero = {0,0,0,0};
  for (int col = startCol; col < K; col += strideCol) {
    for (int row4 = startRow4; row4 < outputFloat4PerCol; row4 += strideRow4) {
    int inIndex  = col * inputFloat4PerCol  + row4;
    int outIndex = col * outputFloat4PerCol + row4;

    outputF4[outIndex] = row4 < inputFloat4PerCol ? inputF4[inIndex] : zero;
    }
  }
}



int main(int argc, char** argv){
    if(argc < 7){
      printf("Usage: %s bs dimX dimY N K ntest\n", argv[0]);
      return 1;
    }
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, 
            *FFT_input, *dFFT_input, *dFFT_output, 
            *iFFT_output, *diFFT_output, *iFFT_output_ref, *diFFT_output_ref;
    long long int bs, dimX, dimY, M, N, K, FFT_len, FFT_bs, iFFT_bs, FFT_input_size, iFFT_output_size;
      bs = atoi(argv[1]);
      dimX = atoi(argv[2]);
      dimY = atoi(argv[3]);
      N = atoi(argv[4]);
      K = atoi(argv[5]);
      M = bs * dimX * THREADBLOCK_M;
      FFT_len = dimY;
      FFT_bs = bs * dimX * K;
      iFFT_bs = bs * dimX * N;
      FFT_input_size = bs * dimX * dimY * K;
      iFFT_output_size = bs * dimX * dimY * N;
      printf("bs=%d dimX=%d dimY=%d M=%d, N=%d K=%d\n", bs, dimX, dimY, M, N, K);
      printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      int ntest = atoi(argv[6]);

      long long int A_size = M * K;
      long long int B_size = N * K;
      long long int C_size = M * N;
      FFT_input = (DataT*)malloc(sizeof(DataT) * (FFT_input_size + ntest));
      iFFT_output = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size + ntest));
      iFFT_output_ref = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size + ntest));
      B = (DataT*)malloc(sizeof(DataT) * (B_size + ntest));
      C = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));
      C_ref = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));


      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_input, sizeof(DataT) * (FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_output, sizeof(DataT) * (FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&diFFT_output, sizeof(DataT) * (iFFT_output_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&diFFT_output_ref, sizeof(DataT) * (iFFT_output_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dA, sizeof(DataT) * (A_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dB, sizeof(DataT) * (B_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC, sizeof(DataT) * (C_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC_ref, sizeof(DataT) * (C_size + ntest)));

      generate_random_vector((float*)FFT_input, FFT_input_size * 2);
      generate_random_vector((float*)B, B_size * 2);
      fill_vector((float*)C, 0, C_size * 2);
      fill_vector((float*)iFFT_output, 0, iFFT_output_size * 2);

      CUDA_RT_CALL(cudaMemcpy(dFFT_input, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dFFT_output, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice));

      CUDA_RT_CALL(cudaMemcpy(diFFT_output, iFFT_output, sizeof(DataT) * iFFT_output_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(diFFT_output_ref, iFFT_output, sizeof(DataT) * iFFT_output_size, cudaMemcpyHostToDevice));
      
      CUDA_RT_CALL(cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));

      DataT alpha = {1.0, -1.0} , beta = {-1.0, 1.0}; 
      // DataT alpha = {1.0, 0} , beta = {1.0, 0}; 

      
      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N + dimY) * THREADBLOCK_K ;  
      printf("*********fused gemm kernel param**********\n");
      printf("blockDim .x=%d .y=%d .z=%d\n", blockDim.x, blockDim.y, blockDim.z);
      printf("gridDim .x=%d .y=%d .z=%d\n", gridDim.x, gridDim.y, gridDim.z);
      printf("shmem size = %d byte\n", shmem_size);
      printf("******************************************\n");
      
      dim3 blockDim_copy(THREADBLOCK_M / 2, 32, 1);

      // The grid:
      //  - x-dim covers the range of row4 in [0, outputFloat4PerCol)
      //  - y-dim covers the range of columns in [0, K)
      dim3 gridDim_copy((THREADBLOCK_M / 2 + blockDim_copy.x - 1) / blockDim_copy.x,
                   (FFT_bs + blockDim_copy.y - 1) / blockDim_copy.y);
      gridDim_copy.x = gridDim_copy.x > 2048 ? 2048 : gridDim_copy.x;
      gridDim_copy.y = gridDim_copy.y > 2048 ? 2048 : gridDim_copy.y;

      printf("************copy kernel param*************\n");
      printf("blockDim .x=%d .y=%d .z=%d\n", blockDim_copy.x, blockDim_copy.y, blockDim_copy.z);
      printf("gridDim .x=%d .y=%d .z=%d\n", gridDim_copy.x, gridDim_copy.y, gridDim_copy.z);
      // printf("shmem size = %d byte\n", shmem_size);
      printf("******************************************\n\n");
      

      cublasHandle_t handle;   
      cublasCreate(&handle);

      cufftHandle plan, iplan;
      CUFFT_CALL(cufftCreate(&plan));
      CUFFT_CALL(cufftCreate(&iplan));
  
      CUFFT_CALL(cufftPlan1d(&plan, FFT_len, CUFFT_C2C, FFT_bs));
      CUFFT_CALL(cufftPlan1d(&iplan, FFT_len, CUFFT_C2C, iFFT_bs));

      printf("***************Verification starts*****************\n");
      printf("start cuFFT!\n");
      CUFFT_CALL(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
                          reinterpret_cast<cufftComplex*>(dFFT_output), 
                          CUFFT_FORWARD));
      cudaDeviceSynchronize();
      printf("start truncation!\n");
      direct_copy_colmajor_float4_truncation<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, FFT_len, FFT_bs, THREADBLOCK_M);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      printf("start cublasCGEMM!\n");
      CUBLAS_CALL(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M));
      cudaDeviceSynchronize();
      printf("start zero-padding!\n");
      direct_copy_colmajor_float4_zero_padding<<<gridDim_copy, blockDim_copy>>>(dC_ref, diFFT_output, FFT_len, iFFT_bs, THREADBLOCK_M);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      printf("start inverse cuFFT!\n");
      CUFFT_CALL(cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(diFFT_output), reinterpret_cast<cufftComplex*>(diFFT_output_ref), CUFFT_FORWARD));
      
      cudaDeviceSynchronize();

            
      printf("Start Fused!\n");
      fused_fft_cgemm_ifft<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_input, dB, dC,diFFT_output, alpha, beta);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      printf("Finish Fused!\n");
        
    

      CUDA_RT_CALL(cudaMemcpy(iFFT_output, diFFT_output, sizeof(DataT) * iFFT_output_size, cudaMemcpyDeviceToHost));
      CUDA_RT_CALL(cudaMemcpy(iFFT_output_ref, diFFT_output_ref, sizeof(DataT) * iFFT_output_size, cudaMemcpyDeviceToHost));
      printf("Compare cuFFT-->DirectCopy-->CGEMM-->Zero Padding Copy-->cuFFT vs. fusedFFT-GEMM!\n");
      verify_vector((float*)iFFT_output_ref, (float*)iFFT_output, iFFT_output_size * 2, dimY);
      printf("***************Finish Verification*****************\n\n");  
      
      printf("***************Profiling starts *****************\n");  

      {
      cudaEvent_t fft_begin, fft_end;
      float elapsed_time;
      cudaEventCreate(&fft_begin);
      cudaEventCreate(&fft_end);

      cudaEventRecord(fft_begin);
      for (int i = 0; i < ntest; ++i){
        cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
        reinterpret_cast<cufftComplex*>(dFFT_output), 
        CUFFT_FORWARD);
        cudaDeviceSynchronize();
        direct_copy_colmajor_float4_truncation<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, FFT_len, FFT_bs, THREADBLOCK_M);
        cudaDeviceSynchronize();
        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);     
        cudaDeviceSynchronize();
        direct_copy_colmajor_float4_zero_padding<<<gridDim_copy, blockDim_copy>>>(dC_ref, diFFT_output, FFT_len, iFFT_bs, THREADBLOCK_M);
        cudaDeviceSynchronize();
        cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(diFFT_output), 
        reinterpret_cast<cufftComplex*>(diFFT_output_ref), 
        CUFFT_FORWARD);

      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      // printf("bs=%d dimX=%d dimY=%d M=%d, N=%d K=%d\n", bs, dimX, dimY, M, N, K);
      // printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      printf("cuFFT-->DirectCopy-->CGEMM-->Zero Padding Copy-->cuFFT, TIME=%8.3f ms\n",  elapsed_time);
    }


      {
        cudaEvent_t fft_begin, fft_end;
        float elapsed_time;
        cudaEventCreate(&fft_begin);
        cudaEventCreate(&fft_end);
      cudaEventRecord(fft_begin);
      for (int i = 0; i < ntest; ++i){
        fused_fft_cgemm_ifft<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_input, dB, dC, diFFT_output, alpha, beta);
        cudaDeviceSynchronize();
      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      // printf("bs=%d dimX=%d dimY=%d M=%d, N=%d K=%d\n", bs, dimX, dimY, M, N, K);
      // printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      printf("fusedFFT-GEMM, TIME=%8.3f ms\n",  elapsed_time);
    }
    return 0;
}