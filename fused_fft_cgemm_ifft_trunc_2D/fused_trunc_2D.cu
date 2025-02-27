
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "fused_fft_cgemm_ifft_trunc_2D.cuh"
#include "fft_radix_2_logN_7_upload_0_stride.cuh"
#include "ifft_radix_2_logN_7_upload_0_stride.cuh"
#include <cufftXt.h>

using DataT = float2;
int thread_bs[4] = {8, 16, 8, 16};

__global__ void direct_copy_colmajor_float4_truncation_2d(
    const float2 *input,
    float2       *output,
    int DY, int DX, int BS, int dimY, int dimX)
{
  const float4* inputF4  = reinterpret_cast<const float4*>(input);
  float4*       outputF4 = reinterpret_cast<float4*>(output);

  int inputFloat4PerCol  = DY / 2;    // total float4 chunks per column
  int outputFloat4PerCol = dimY / 2;  // truncated float4 chunks

  int startRow4 = blockIdx.x * blockDim.x + threadIdx.x; 
  int startCol  = blockIdx.y * blockDim.y + threadIdx.y;

  int strideRow4 = blockDim.x * gridDim.x;
  int strideCol  = blockDim.y * gridDim.y;

  for (int col = startCol; col < DX * BS; col += strideCol) {
    // check if the X dimension is within dimX
    if ((col % DX) < dimX) {
      for (int row4 = startRow4; row4 < outputFloat4PerCol; row4 += strideRow4) {
        int inIndex  = col * inputFloat4PerCol  + row4;
        int outIndex = ((col / DX) * dimX + col % DX)* outputFloat4PerCol + row4;
        // copy 2 float2 at once
        outputF4[outIndex] = inputF4[inIndex];
      }
    }
  }
}
// __global__ void direct_copy_colmajor_float4_truncation_2d(const float2 *input,
//   float2       *output,
//   int DY, int DX, int BS, int dimY, int dimX)
// {
//   const float4 *inputF4  = reinterpret_cast<const float4*>(input);
//   float4       *outputF4 = reinterpret_cast<float4*>(output);

//   int inputFloat4PerCol  = DY / 2;         // total float4 chunks in each column
//   int outputFloat4PerCol = dimY / 2; // float4 chunks in truncated top rows

//   int startRow4 = blockIdx.x * blockDim.x + threadIdx.x; 
//   int startCol  = blockIdx.y * blockDim.y + threadIdx.y;

//   int strideRow4 = blockDim.x * gridDim.x;
//   int strideCol  = blockDim.y * gridDim.y;

//   for (int col = startCol; col < DX * BS; col += strideCol) {
//     if((col % DX ) < dimX){
//       for (int row4 = startRow4; row4 < outputFloat4PerCol; row4 += strideRow4) {
//       int inIndex  = col * inputFloat4PerCol  + row4;
//       int outIndex = (col / DX) * dimX * outputFloat4PerCol + row4;
//       outputF4[outIndex] = inputF4[inIndex];
//       }
//     }
//   }
// }

__global__ void direct_copy_colmajor_float4_zero_padding_2d(const float2 *input,
  float2       *output,
  int DY, int DX, int BS, int dimY, int dimX)
{
  const float4 *inputF4  = reinterpret_cast<const float4*>(input);
  float4       *outputF4 = reinterpret_cast<float4*>(output);

  int inputFloat4PerCol  = dimY / 2;         // total float4 chunks in each column
  int outputFloat4PerCol = DY / 2; // float4 chunks in truncated top rows

  int startRow4 = blockIdx.x * blockDim.x + threadIdx.x; 
  int startCol  = blockIdx.y * blockDim.y + threadIdx.y;

  int strideRow4 = blockDim.x * gridDim.x;
  int strideCol  = blockDim.y * gridDim.y;
  float4 zero = {0,0,0,0};


  for (int col = startCol; col < DX * BS; col += strideCol) {
      for (int row4 = startRow4; row4 < outputFloat4PerCol; row4 += strideRow4) {
      int inIndex  = ((col / DX) * dimX + col % DX) * inputFloat4PerCol  + row4;
      int outIndex = col * outputFloat4PerCol + row4;
      outputF4[outIndex] = (row4 < inputFloat4PerCol && (col % DX ) < dimX) ? inputF4[inIndex] : zero;
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
    long long int bs, dimX, dimY, DX, DY, M, N, K, FFT_len, FFT_bs, iFFT_bs, FFT_input_size, iFFT_output_size;
      dimX = 64; 
      dimY = 64;  
      bs = atoi(argv[1]);
      DX = atoi(argv[2]);
      DY = atoi(argv[3]);
      N = atoi(argv[4]);
      K = atoi(argv[5]);
      int ntest = atoi(argv[6]);
      int threadblock_bs = argc > 7 ? atoi(argv[7]) : 4;
      M = bs * dimX * THREADBLOCK_M;
      FFT_len = dimY;
      FFT_bs = bs * dimX * K;
      iFFT_bs = bs * dimX * N;
      int FFT_bs_2d = bs * K;
      int iFFT_bs_2d = bs * N;
      FFT_input_size = bs * DX * DY * K;
      iFFT_output_size = bs * DX * DY * N;
      printf("bs=%d DX=%d DY=%d M=%d, N=%d K=%d\n", bs, DX, DY, M, N, K);
      printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);


      long long int A_size = M * K;
      long long int B_size = N * K;
      long long int C_size = M * N;
      FFT_input = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size > FFT_input_size ? iFFT_output_size : FFT_input_size + ntest));
      iFFT_output = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size + ntest));
      iFFT_output_ref = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size + ntest));
      B = (DataT*)malloc(sizeof(DataT) * (B_size + ntest));
      C = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));
      C_ref = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));


      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_input, sizeof(DataT) * (iFFT_output_size > FFT_input_size ? iFFT_output_size : FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_output, sizeof(DataT) * (iFFT_output_size > FFT_input_size ? iFFT_output_size : FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&diFFT_output, sizeof(DataT) * (iFFT_output_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&diFFT_output_ref, sizeof(DataT) * (iFFT_output_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dA, sizeof(DataT) * (A_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dB, sizeof(DataT) * (B_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC, sizeof(DataT) * (C_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC_ref, sizeof(DataT) * (C_size + ntest)));

      generate_random_vector((float*)FFT_input, FFT_input_size * 2);
      generate_random_vector((float*)B, B_size * 2);
      // fill_vector((float*)FFT_input, 0, FFT_input_size * 2);
      // fill_vector((float*)B, 0, B_size * 2);
      fill_vector((float*)C, 0, C_size * 2);
      fill_vector((float*)iFFT_output, 0, iFFT_output_size * 2);
      // FFT_input[0] = {1, 1};
      // B[0] = {1, 1};
      CUDA_RT_CALL(cudaMemcpy(dFFT_input, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice));
      // CUDA_RT_CALL(cudaMemcpy(dFFT_output, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice));

      CUDA_RT_CALL(cudaMemcpy(diFFT_output, iFFT_output, sizeof(DataT) * iFFT_output_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(diFFT_output_ref, iFFT_output, sizeof(DataT) * iFFT_output_size, cudaMemcpyHostToDevice));
      
      CUDA_RT_CALL(cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));

      DataT alpha = {1.0, -1.0} , beta = {-1.0, 1.0}; 
      // DataT alpha = {1.0, 0} , beta = {1.0, 0}; 

      
      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N + DY) * THREADBLOCK_K ;  
      

      
      dim3 blockDim_copy(32, 32, 1);

      // The grid:
      //  - x-dim covers the range of row4 in [0, outputFloat4PerCol)
      //  - y-dim covers the range of columns in [0, K)
      dim3 gridDim_copy((32 + blockDim_copy.x - 1) / blockDim_copy.x,
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
    
      int n[2] = {DY, DX};  // 2D FFT 维度
      CUFFT_CALL(cufftPlanMany(&plan, 2, n,
        nullptr, 1, DY * DX,   // 输入紧密存储
        nullptr, 1, DY * DX,   // 输出紧密存储
        CUFFT_C2C, FFT_bs_2d)); // 批处理数量
      CUFFT_CALL(cufftPlanMany(&iplan, 2, n,
        nullptr, 1, DY * DX,   // 输入紧密存储
        nullptr, 1, DY * DX,   // 输出紧密存储
        CUFFT_C2C, iFFT_bs_2d)); // 批处理数量

      printf("***************Verification starts*****************\n");
      printf("start cuFFT!\n");
      CUFFT_CALL(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
                          reinterpret_cast<cufftComplex*>(dFFT_output), 
                          CUFFT_FORWARD));
      cudaDeviceSynchronize();
      printf("start truncation!\n");
      direct_copy_colmajor_float4_truncation_2d<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, DY, DX, bs*K, dimY, dimX);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      printf("start cublasCGEMM!\n");
      CUBLAS_CALL(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M));
      cudaDeviceSynchronize();
      printf("start zero-padding!\n");
      direct_copy_colmajor_float4_zero_padding_2d<<<gridDim_copy, blockDim_copy>>>(dC_ref, diFFT_output, DY, DX, bs*N, dimY, dimX);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      CHECK_CUDA_KERNEL();
      printf("start inverse cuFFT!\n");
      CUFFT_CALL(cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(diFFT_output), reinterpret_cast<cufftComplex*>(diFFT_output_ref), CUFFT_FORWARD));
      
      cudaDeviceSynchronize();
      



      
      dim3 gridDim_fft_dimx((DY * K * bs + threadblock_bs - 1) / threadblock_bs, 1, 1);
      dim3 gridDim_ifft_dimx((DY * N * bs + threadblock_bs - 1) / threadblock_bs, 1, 1);
      gridDim_fft_dimx.x = gridDim_fft_dimx.x > 65536 ? 65536 : gridDim_fft_dimx.x;
      gridDim_ifft_dimx.x = gridDim_ifft_dimx.x > 65536 ? 65536 : gridDim_ifft_dimx.x;
      dim3 blockDim_fft_dimx(DX / thread_bs[0] * threadblock_bs, 1, 1); 
      int shmem_size_fft_dimx = sizeof(DataT) * DY * threadblock_bs ;  
      
      printf("Start X-dim FFT\n\n");
      printf("********* DX-dim FFT**********\n");
      printf("blockDim .x=%d .y=%d .z=%d\n", blockDim_fft_dimx.x, blockDim_fft_dimx.y, blockDim_fft_dimx.z);
      printf("gridDim .x=%d .y=%d .z=%d\n", gridDim_fft_dimx.x, gridDim_fft_dimx.y, gridDim_fft_dimx.z);
      printf("shmem size = %d byte\n", shmem_size);
      printf("******************************************\n");
      fft_7_stride<<<gridDim_fft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, dFFT_output, threadblock_bs, DY, DY * K * bs);
      CHECK_CUDA_KERNEL();
      printf("Start Fused!\n");
      printf("*********fused gemm kernel param**********\n");
      printf("blockDim .x=%d .y=%d .z=%d\n", blockDim.x, blockDim.y, blockDim.z);
      printf("gridDim .x=%d .y=%d .z=%d\n", gridDim.x, gridDim.y, gridDim.z);
      printf("shmem size = %d byte\n", shmem_size);
      printf("******************************************\n");
      fused_fft_cgemm_ifft<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_output, dB, dC,dFFT_input, alpha, beta);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      printf("Start X-dim iFFT\n");
      printf("Start X-dim FFT\n\n");
      printf("********* DX-dim FFT**********\n");
      printf("blockDim .x=%d .y=%d .z=%d\n", blockDim_fft_dimx.x, blockDim_fft_dimx.y, blockDim_fft_dimx.z);
      printf("gridDim .x=%d .y=%d .z=%d\n", gridDim_ifft_dimx.x, gridDim_ifft_dimx.y, gridDim_ifft_dimx.z);
      printf("shmem size = %d byte\n", shmem_size);
      printf("******************************************\n");
      ifft_7_stride<<<gridDim_ifft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, diFFT_output, threadblock_bs, DY, DY * N * bs);
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
        direct_copy_colmajor_float4_truncation_2d<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, DY, DX, bs*K, dimY, dimX);
        
        cudaDeviceSynchronize();
        
        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);
        cudaDeviceSynchronize();
        direct_copy_colmajor_float4_zero_padding_2d<<<gridDim_copy, blockDim_copy>>>(dC_ref, diFFT_output, DY, DX, bs*N, dimY, dimX);
        cudaDeviceSynchronize();
        cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(diFFT_output), reinterpret_cast<cufftComplex*>(diFFT_output_ref), CUFFT_FORWARD);
        cudaDeviceSynchronize();
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
        fft_7_stride<<<gridDim_fft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, dFFT_output, threadblock_bs, DY, DY * K * bs);
        cudaDeviceSynchronize();
        fused_fft_cgemm_ifft<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_output, dB, dC,dFFT_input, alpha, beta);
        cudaDeviceSynchronize();
        ifft_7_stride<<<gridDim_ifft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, diFFT_output, threadblock_bs, DY, DY * N * bs);
        cudaDeviceSynchronize();
      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      // printf("bs=%d dimX=%d dimY=%d M=%d, N=%d K=%d\n", bs, dimX, dimY, M, N, K);
      // printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      printf("fusedFFT-GEMM,                                          TIME=%8.3f ms\n",  elapsed_time);
    }
    return 0;
}