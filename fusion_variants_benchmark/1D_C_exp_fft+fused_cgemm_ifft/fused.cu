
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "TurboFNO.h"
#include "fused_cgemm_ifft_7.cuh"
#include "fused_cgemm_ifft_8.cuh"
#include "fused_cgemm_ifft_9.cuh"
#include "fused_cgemm_ifft_10.cuh"
#include "fft_radix_2_logN_7_upload_0_stride_DY.cuh"
#include "fft_radix_2_logN_8_upload_0_stride_DY.cuh"
#include "fft_radix_2_logN_9_upload_0_stride_DY.cuh"
#include "fft_radix_2_logN_10_upload_0_stride_DY.cuh"
#include <cufftXt.h>


using DataT = float2;
int thread_bs[4] = {8, 16, 8, 16};
void (*fused_cgemm_ifft [4])(int, int, int, float2 *, float2 *, float2 *, float2 *, float2, float2) = 
{fused_cgemm_ifft_7, fused_cgemm_ifft_8, fused_cgemm_ifft_9, fused_cgemm_ifft_10};
void (*fft_stride_DY [4])(float2 *, float2 *, int, int) = {fft_7_stride_DY, fft_8_stride_DY, fft_9_stride_DY, fft_10_stride_DY};



int main(int argc, char** argv){
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, 
            *FFT_input, *dFFT_input, *dFFT_output, 
            *iFFT_output, *diFFT_output, *iFFT_output_ref, *diFFT_output_ref;
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
      shmem_size = shmem_size > sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N + DY) * THREADBLOCK_K ? shmem_size : sizeof(DataT)  * (THREADBLOCK_M + THREADBLOCK_N + DY) * THREADBLOCK_K;
      
      
      cudaDeviceSynchronize();
      int logFFT_len = int(log2f(DY)) - 7;
      dim3 gridDim_fft_dimY((dimX * K * bs + threadblock_bs - 1) / threadblock_bs, 1, 1);
      dim3 gridDim_ifft_dimY((dimX * N * bs + threadblock_bs - 1) / threadblock_bs, 1, 1);
      gridDim_fft_dimY.x = gridDim_fft_dimY.x > 65536 ? 65536 : gridDim_fft_dimY.x;
      gridDim_ifft_dimY.x = gridDim_ifft_dimY.x > 65536 ? 65536 : gridDim_ifft_dimY.x;
      dim3 blockDim_fft_dimY(DY / thread_bs[logFFT_len] * threadblock_bs, 1, 1); 
      int shmem_size_fft_dimY = sizeof(DataT) * DY * threadblock_bs ;  
      
      fft_stride_DY[logFFT_len]<<<gridDim_fft_dimY, blockDim_fft_dimY, shmem_size_fft_dimY>>>(dFFT_input,  dA, threadblock_bs, dimX * K * bs);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      CHECK_CUDA_KERNEL();
      
      fused_cgemm_ifft[logFFT_len]<<<gridDim, blockDim, shmem_size>>>(M, N, K, dA, dB, dC, diFFT_output, alpha, beta);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      CHECK_CUDA_KERNEL();


      {
        cudaEvent_t fft_begin, fft_end;
        float elapsed_time;
        cudaEventCreate(&fft_begin);
        cudaEventCreate(&fft_end);
      cudaEventRecord(fft_begin);
      for (int i = 0; i < ntest; ++i){
        fft_stride_DY[logFFT_len]<<<gridDim_fft_dimY, blockDim_fft_dimY, shmem_size_fft_dimY>>>(dFFT_input, dA, threadblock_bs, dimX * K * bs);
        cudaDeviceSynchronize();
        fused_cgemm_ifft[logFFT_len]<<<gridDim, blockDim, shmem_size>>>(M, N, K, dA, dB, dC, diFFT_output, alpha, beta);
        cudaDeviceSynchronize();
      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      printf("1D_C, bs=%-4d, dimX=%-4d, DY=%-4d, N=%-4d, K=%-4d, TIME=%8.3fms\n",
        bs, dimX, DY, N, K, elapsed_time);
    }
  // }}}}}
    return 0;
}