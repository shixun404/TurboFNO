
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "fused_fft_cgemm.cuh"
#include <cufftXt.h>

using DataT = float2;
int thread_bs[4] = {8, 16, 8, 16};

// Kernel to truncate the top 'reduced_M' rows of an (M x K) column-major matrix of float2
// by copying them in chunks of float4 (2 x float2). 
__global__ void direct_copy_colmajor_float4(const float2 *input, 
  float2       *output,
  int M, int K, int reduced_M)
{
// Reinterpret float2* as float4* (each float4 = 2 float2)
const float4 *inputF4  = reinterpret_cast<const float4 *>(input);
float4       *outputF4 = reinterpret_cast<float4       *>(output);

// Number of float4 elements per column in the *original* M rows
int inputFloat4PerCol  = M / 2; 
// Number of float4 elements per column for the truncated (reduced_M) rows
int outputFloat4PerCol = reduced_M / 2;

// 2D block & grid:
//   - 'col' indexes the columns (up to K)
//   - 'row4' indexes the float4 chunks along the row dimension (up to reduced_M/2)
int col  = blockIdx.y * blockDim.y + threadIdx.y;  // column index [0..K-1]
int row4 = blockIdx.x * blockDim.x + threadIdx.x;  // float4 index [0..(reduced_M/2)-1]

// Bounds check
if (col < K && row4 < outputFloat4PerCol) {
// In column-major, the float4 index for the input is:
//      inIndex  = col * (M/2) + row4
// For the output (which has reduced_M rows):
//      outIndex = col * (reduced_M/2) + row4

int inIndex  = col * inputFloat4PerCol  + row4;
int outIndex = col * outputFloat4PerCol + row4;

// Copy one float4 (which is 2 float2)
outputF4[outIndex] = inputF4[inIndex];
}
}


int main(int argc, char** argv){
    if(argc < 7){
      printf("Usage: %s bs dimX dimY N K ntest\n", argv[0]);
      return 1;
    }
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, *FFT_input, *dFFT_input, *FFT_output, *dFFT_output;
    int bs, dimX, dimY, M, N, K, FFT_len, FFT_bs, FFT_input_size;
      bs = atoi(argv[1]);
      dimX = atoi(argv[2]);
      dimY = atoi(argv[3]);
      N = atoi(argv[4]);
      K = atoi(argv[5]);
      M = bs * dimX * THREADBLOCK_M;
      FFT_len = dimY;
      FFT_bs = bs * dimX * K;
      FFT_input_size = bs * dimX * dimY * K;
      printf("bs=%d dimX=%d dimY=%d M=%d, N=%d K=%d\n", bs, dimX, dimY, M, N, K);
      printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      int num_tests = atoi(argv[6]);

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

      generate_random_vector((float*)FFT_input, FFT_input_size * 2);
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
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N + dimY) * THREADBLOCK_K ;  
      printf("blockDim .x=%d .y=%d .z=%d\n", blockDim.x, blockDim.y, blockDim.z);
      printf("gridDim .x=%d .y=%d .z=%d\n", gridDim.x, gridDim.y, gridDim.z);
      printf("shmem size = %d byte\n", shmem_size);
      dim3 blockDim_copy(256, 1, 1);

      // The grid:
      //  - x-dim covers the range of row4 in [0, outputFloat4PerCol)
      //  - y-dim covers the range of columns in [0, K)
      dim3 gridDim_copy((THREADBLOCK_M / 2 + blockDim_copy.x - 1) / blockDim_copy.x,
                   (FFT_bs + blockDim_copy.y - 1) / blockDim_copy.y);

      

      cublasHandle_t handle;   
      cublasCreate(&handle);

      cufftHandle plan;
      cufftCreate(&plan);
  
      cufftPlan1d(&plan, FFT_len, CUFFT_C2C, FFT_bs);

      printf("start cuFFT!\n");
      cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
                          reinterpret_cast<cufftComplex*>(dFFT_output), 
                          CUFFT_FORWARD);
      cudaDeviceSynchronize();
      printf("start copy!\n");
      fflush(stdout);
      direct_copy_colmajor_float4<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, FFT_len, FFT_bs, THREADBLOCK_M);
      cudaError_t err = cudaGetLastError();  // 获取最近的错误
      if (err != cudaSuccess) {
          printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
      }
      cudaDeviceSynchronize();
      printf("Finish copy!\n");
      err = cudaGetLastError();  // 获取最近的错误
      if (err != cudaSuccess) {
          printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
      }
      cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);     
      cudaDeviceSynchronize();


      printf("Start Fused!\n");
      fused_fft_cgemm<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_input, dB, dC, alpha, beta);
      
      err = cudaGetLastError();  // 获取最近的错误
      if (err != cudaSuccess) {
          printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
      }
      cudaDeviceSynchronize();
      printf("Finish Fused!\n");
      err = cudaGetLastError();  // 获取最近的错误
      if (err != cudaSuccess) {
          printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
      }
        
    

      cudaMemcpy(C, dC, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);

      verify_vector((float*)C_ref, (float*)C, M * N * 2, M);

    return 0;
}