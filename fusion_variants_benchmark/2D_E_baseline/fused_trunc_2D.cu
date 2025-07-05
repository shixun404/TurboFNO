
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "TurboFNO.h"

#include <cufftXt.h>

using DataT = float2;


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
   
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, 
            *FFT_input, *dFFT_input, *dFFT_output;
            long long int bs, dimX, dimY, DX, DY, M, N, K, FFT_len, FFT_bs, iFFT_bs, FFT_input_size, iFFT_output_size, FFT_bs_2d, iFFT_bs_2d;
            dimX = 64; 
            dimY = 64;  
            bs = 128;
            DX = 256;
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
            FFT_len = DY;
            FFT_bs = bs * dimX * K;
            iFFT_bs = bs * dimX * N;
            FFT_input_size = bs * dimX * DY * K;
            iFFT_output_size = bs * dimX * DY * N;
            M = bs * dimX * THREADBLOCK_M;
            FFT_len = dimY;
            FFT_bs = bs * dimX * K;
            iFFT_bs = bs * dimX * N;
       
            FFT_bs_2d = bs * K;
            iFFT_bs_2d = bs * N;
            
            FFT_input_size = bs * DX * DY * K;
            iFFT_output_size = bs * DX * DY * N;


      long long int A_size = M * K;
      long long int B_size = N * K;
      long long int C_size = M * N;
      FFT_input = (DataT*)malloc(sizeof(DataT) * (iFFT_output_size > FFT_input_size ? iFFT_output_size : FFT_input_size + ntest));
      B = (DataT*)malloc(sizeof(DataT) * (B_size + ntest));
      C = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));
      C_ref = (DataT*)malloc(sizeof(DataT) * (C_size + ntest));


      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_input, sizeof(DataT) * (iFFT_output_size > FFT_input_size ? iFFT_output_size : FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dFFT_output, sizeof(DataT) * (iFFT_output_size > FFT_input_size ? iFFT_output_size : FFT_input_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dA, sizeof(DataT) * (A_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dB, sizeof(DataT) * (B_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC, sizeof(DataT) * (C_size + ntest)));
      CUDA_RT_CALL(cudaMalloc((void**)&dC_ref, sizeof(DataT) * (C_size + ntest)));

      generate_random_vector((float*)FFT_input, FFT_input_size * 2);
      generate_random_vector((float*)B, B_size * 2);
      // fill_vector((float*)FFT_input, 0, FFT_input_size * 2);
      // fill_vector((float*)B, 0, B_size * 2);
      fill_vector((float*)C, 0, C_size * 2);
      // FFT_input[0] = {1, 1};
      // B[0] = {1, 1};
      CUDA_RT_CALL(cudaMemcpy(dFFT_input, FFT_input, sizeof(DataT) * FFT_input_size, cudaMemcpyHostToDevice));      
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
      // auto& DX_list = config["DX_list"];
      // auto& DY_list   = config["DY_list"];
      // auto& N_list    = config["N_list"];
      // auto& K_list    = config["K_list"];
           
      // for (int bs : bs_list) {
      //   for (int DX : DX_list) {
      //       for (int DY : DY_list) {
      //           for (int N : N_list) {
      //               for (int K : K_list) {

                
                      M = bs * dimX * THREADBLOCK_M;
                      FFT_len = DY;
                      FFT_bs = bs * dimX * K;
                      iFFT_bs = bs * dimX * N;
                      FFT_input_size = bs * dimX * DY * K;
                      iFFT_output_size = bs * dimX * DY * N;
                      M = bs * dimX * THREADBLOCK_M;
                      FFT_len = dimY;
                      FFT_bs = bs * dimX * K;
                      iFFT_bs = bs * dimX * N;
                 
                      FFT_bs_2d = bs * K;
                      iFFT_bs_2d = bs * N;
                      
                      FFT_input_size = bs * DX * DY * K;
                      iFFT_output_size = bs * DX * DY * N;
      
      dim3 blockDim_copy(32, 32, 1);

      // The grid:
      //  - x-dim covers the range of row4 in [0, outputFloat4PerCol)
      //  - y-dim covers the range of columns in [0, K)
      dim3 gridDim_copy((32 + blockDim_copy.x - 1) / blockDim_copy.x,
                   (FFT_bs + blockDim_copy.y - 1) / blockDim_copy.y);
      gridDim_copy.x = gridDim_copy.x > 2048 ? 2048 : gridDim_copy.x;
      gridDim_copy.y = gridDim_copy.y > 2048 ? 2048 : gridDim_copy.y;


      cublasHandle_t handle;   
      cublasCreate(&handle);

      cufftHandle plan, iplan;
      CUFFT_CALL(cufftCreate(&plan));
      CUFFT_CALL(cufftCreate(&iplan));
    
      int n[2] = {DX, DY};  // 2D FFT 维度
      CUFFT_CALL(cufftPlanMany(&plan, 2, n,
        nullptr, 1, DY * DX,   // 输入紧密存储
        nullptr, 1, DY * DX,   // 输出紧密存储
        CUFFT_C2C, FFT_bs_2d)); // 批处理数量
      CUFFT_CALL(cufftPlanMany(&iplan, 2, n,
        nullptr, 1, DY * DX,   // 输入紧密存储
        nullptr, 1, DY * DX,   // 输出紧密存储
        CUFFT_C2C, iFFT_bs_2d)); // 批处理数量

      CUFFT_CALL(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dFFT_input), 
                          reinterpret_cast<cufftComplex*>(dFFT_output), 
                          CUFFT_FORWARD));
      cudaDeviceSynchronize();
    
      direct_copy_colmajor_float4_truncation_2d<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, DY, DX, bs*K, dimY, dimX);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
    
      CUBLAS_CALL(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M));
      cudaDeviceSynchronize();
    
      direct_copy_colmajor_float4_zero_padding_2d<<<gridDim_copy, blockDim_copy>>>(dC_ref, dFFT_output, DY, DX, bs*N, dimY, dimX);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      CHECK_CUDA_KERNEL();
    
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
        direct_copy_colmajor_float4_truncation_2d<<<gridDim_copy, blockDim_copy>>>(dFFT_output, dA, DY, DX, bs*K, dimY, dimX);
        
        cudaDeviceSynchronize();
        
        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);
        cudaDeviceSynchronize();
        direct_copy_colmajor_float4_zero_padding_2d<<<gridDim_copy, blockDim_copy>>>(dC_ref, dFFT_output, DY, DX, bs*N, dimY, dimX);
        cudaDeviceSynchronize();
        cufftExecC2C(iplan, reinterpret_cast<cufftComplex*>(dFFT_output), reinterpret_cast<cufftComplex*>(dFFT_input), CUFFT_FORWARD);
        cudaDeviceSynchronize();
      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      printf("1D_E, bs=%-4d, DX=%-4d, DY=%-4d, N=%-4d, K=%-4d, TIME=%8.3fms\n",
        bs, DX, DY, N, K, elapsed_time);
    }
    CUFFT_CALL(cufftDestroy(plan));
    CUFFT_CALL(cufftDestroy(iplan));
    cudaDeviceSynchronize();
  // }}}}}

    return 0;
}