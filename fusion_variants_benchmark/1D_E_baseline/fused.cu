
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"

#include "TurboFNO.h"
#include <cufftXt.h>
#include <vector>


using DataT = float2;
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
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, 
            *FFT_input, *dFFT_input, *dFFT_output, 
            *iFFT_output, *iFFT_output_ref;
    long long int bs, dimX, dimY, DY, M, N, K, FFT_len, FFT_bs, iFFT_bs, FFT_input_size, iFFT_output_size;

      bs = 128;
      dimX = 256;
      DY = 256;
      N =  128;
      K = 128;
      ntest = 5;

          // 解析命令行参数
    if (argc > 1) {
      if (argc != 6) {
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
          
      }
  }

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
      FFT_input = (DataT*)malloc(sizeof(DataT) * (FFT_input_size + ntest));
      iFFT_output = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size + ntest));
      iFFT_output_ref = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size + ntest));
      B = (DataT*)malloc(sizeof(DataT) * (B_size + ntest));
      C = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));
      C_ref = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));


      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_input, sizeof(DataT) * (FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_output, sizeof(DataT) * (FFT_input_size + ntest)));
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

      CUDA_RT_CALL(cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice));

      DataT alpha = {1.0, -1.0} , beta = {-1.0, 1.0}; 
      // DataT alpha = {1.0, 0} , beta = {1.0, 0}; 

            
      // std::ifstream infile(DEFAULT_CONFIG_PATH );
      // std::string line;
  
      // std::unordered_map<std::string, std::vector<int>> config;
  
      // while (std::getline(infile, line)) {
      //     if (line.empty() || line[0] == '#') continue;  // 跳过注释或空行
      //     std::istringstream iss(line);
      //     std::string key;
      //     iss >> key;
      //     config[key] = parse_line(line);
      // }
  
      // // 提取参数
      // auto& bs_list   = config["bs_list"];
      // auto& dimX_list = config["dimX_list"];
      // auto& DY_list   = config["DY_list"];
      // auto& N_list    = config["N_list"];
      // auto& K_list    = config["K_list"];
      
      // for (int bs : bs_list) {
      //   for (int dimX : dimX_list) {
      //       for (int DY : DY_list) {
      //           for (int N : N_list) {
      //               for (int K : K_list) {


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
      
      CUDA_RT_CALL(cudaMemcpy(dFFT_input, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice));

                
      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K * 2;

      dim3 blockDim_copy(THREADBLOCK_M / 2, 32, 1);
      dim3 gridDim_copy((THREADBLOCK_M / 2 + blockDim_copy.x - 1) / blockDim_copy.x,
                   (FFT_bs + blockDim_copy.y - 1) / blockDim_copy.y);
      gridDim_copy.x = gridDim_copy.x > 2048 ? 2048 : gridDim_copy.x;
      gridDim_copy.y = gridDim_copy.y > 2048 ? 2048 : gridDim_copy.y;
      cublasHandle_t handle;   
      cublasCreate(&handle);

      cufftHandle plan, iplan;
      CUFFT_CALL(cufftCreate(&plan));
      CUFFT_CALL(cufftCreate(&iplan));
                      
      CUFFT_CALL(cufftPlan1d(&plan, FFT_len, CUFFT_C2C, FFT_bs));
      CUFFT_CALL(cufftPlan1d(&iplan, FFT_len, CUFFT_C2C, iFFT_bs));
      CUFFT_CALL(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
                          reinterpret_cast<cufftComplex*>(dFFT_output), 
                          CUFFT_FORWARD));
      cudaDeviceSynchronize();
      direct_copy_colmajor_float4_truncation<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, FFT_len, FFT_bs, THREADBLOCK_M);
      cudaDeviceSynchronize();
      CUBLAS_CALL(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M));
      cudaDeviceSynchronize();
      direct_copy_colmajor_float4_zero_padding<<<gridDim_copy, blockDim_copy>>>(dC_ref, dFFT_output, FFT_len, iFFT_bs, THREADBLOCK_M);
      cudaDeviceSynchronize();
      CUFFT_CALL(cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(dFFT_output), reinterpret_cast<cufftComplex*>(dFFT_input), CUFFT_FORWARD));
      
      cudaDeviceSynchronize();

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
        direct_copy_colmajor_float4_zero_padding<<<gridDim_copy, blockDim_copy>>>(dC_ref, dFFT_output, FFT_len, iFFT_bs, THREADBLOCK_M);
        cudaDeviceSynchronize();
        cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(dFFT_output), 
        reinterpret_cast<cufftComplex*>(dFFT_input), 
        CUFFT_FORWARD);

      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      // printf("bs=%d dimX=%d DY=%d M=%d, N=%d K=%d\n", bs, dimX, DY, M, N, K);
      // printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      printf("1D_E, bs=%-4d, dimX=%-4d, DY=%-4d, N=%-4d, K=%-4d, TIME=%8.3fms\n",
        bs, dimX, DY, N, K, elapsed_time);
    }
    CUFFT_CALL(cufftDestroy(plan));
    CUFFT_CALL(cufftDestroy(iplan));
    cudaDeviceSynchronize();
//   }
// }
// }
// }
// }
    return 0;
}