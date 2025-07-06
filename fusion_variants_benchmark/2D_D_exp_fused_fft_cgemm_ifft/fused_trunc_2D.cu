
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "TurboFNO.h"
#include "fused_fft_cgemm_ifft_trunc_2D_7.cuh"
#include "fused_fft_cgemm_ifft_trunc_2D_8.cuh"
#include "fused_fft_cgemm_ifft_trunc_2D_9.cuh"
#include "fused_fft_cgemm_ifft_trunc_2D_10.cuh"
#include "fft_radix_2_logN_7_upload_0_stride.cuh"
#include "ifft_radix_2_logN_7_upload_0_stride.cuh"

#include "fft_radix_2_logN_8_upload_0_stride.cuh"
#include "ifft_radix_2_logN_8_upload_0_stride.cuh"

#include "fft_radix_2_logN_9_upload_0_stride.cuh"
#include "ifft_radix_2_logN_9_upload_0_stride.cuh"

#include "fft_radix_2_logN_10_upload_0_stride.cuh"
#include "ifft_radix_2_logN_10_upload_0_stride.cuh"

#include <cufftXt.h>

using DataT = float2;
int thread_bs[4] = {8, 16, 8, 16};
void (*fft_stride [4])(float2 *, float2 *, int, int, int, int) = {fft_7_stride, fft_8_stride, fft_9_stride, fft_10_stride};
void (*ifft_stride [4])(float2 *, float2 *, int, int, int, int) = {ifft_7_stride, ifft_8_stride, ifft_9_stride, ifft_10_stride};
void (*fused_fft_cgemm_ifft [4])(int, int, int, float2 *, float2 *, float2 *, float2 *, float2, float2) = 
{fused_fft_cgemm_ifft_7, fused_fft_cgemm_ifft_8, fused_fft_cgemm_ifft_9, fused_fft_cgemm_ifft_10};


int main(int argc, char** argv){

      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref, 
            *FFT_input, *dFFT_input, *dFFT_output, 
            *iFFT_output, *diFFT_output, *iFFT_output_ref, *diFFT_output_ref;
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
                      printf("Usage: %s <bs> <DX> <DY> <N> <K>\n", argv[0]);
                      printf("Example: %s 128 256 256 128 128\n", argv[0]);
                      printf("Using default values: bs=%lld, DX=%lld, DY=%lld, N=%lld, K=%lld\n", 
                             bs, dimX, DY, N, K);
                  } else {
                      bs = atoi(argv[1]);
                      DX = atoi(argv[2]);
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
      cudaDeviceSynchronize();

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

      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M + THREADBLOCK_N + DY) * THREADBLOCK_K ;  
      dim3 gridDim_fft_dimx((DY * K * bs + threadblock_bs - 1) / threadblock_bs, 1, 1);
      dim3 gridDim_ifft_dimx((DY * N * bs + threadblock_bs - 1) / threadblock_bs, 1, 1);
      long long int num_blocks = (DY * K * bs + threadblock_bs - 1) / threadblock_bs;
      gridDim_fft_dimx.x = num_blocks > 65536 ? 65536 : num_blocks;
      gridDim_ifft_dimx.x = num_blocks > 65536 ? 65536 : num_blocks;
      
      gridDim_fft_dimx.y = (num_blocks + 65535) / 65536 > 1 ? (num_blocks + 65535) / 65536 : 1;
      gridDim_ifft_dimx.y = (num_blocks + 65535) / 65536 > 1 ? (num_blocks + 65535) / 65536 : 1;
      
      
      dim3 blockDim_fft_dimx(DX / thread_bs[int(log2f(DX)) - 7] * threadblock_bs, 1, 1); 
      int shmem_size_fft_dimx = sizeof(DataT) * DX * threadblock_bs ;  
      
      fft_stride[int(log2f(DX)) - 7]<<<gridDim_fft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, dFFT_output, threadblock_bs, DY, DY * K * bs, dimX);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      CHECK_CUDA_KERNEL();
     
      fused_fft_cgemm_ifft[int(log2f(DY)) - 7]<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_output, dB, dC,dFFT_input, alpha, beta);
      CHECK_CUDA_KERNEL();
      cudaDeviceSynchronize();
      CHECK_CUDA_KERNEL();
      
      ifft_stride[int(log2f(DX)) - 7]<<<gridDim_ifft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, diFFT_output, threadblock_bs, DY, DY * N * bs, dimX);
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
        fft_stride[int(log2f(DX)) - 7]<<<gridDim_fft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(dFFT_input, dFFT_output, threadblock_bs, DY, DY * K * bs, dimX);
        cudaDeviceSynchronize();
        fused_fft_cgemm_ifft[int(log2f(DY)) - 7]<<<gridDim, blockDim, shmem_size>>>(M, N, K, dFFT_output, dB, dC, diFFT_output, alpha, beta);
        cudaDeviceSynchronize();
        ifft_stride[int(log2f(DX)) - 7]<<<gridDim_ifft_dimx, blockDim_fft_dimx, shmem_size_fft_dimx>>>(diFFT_output, diFFT_output_ref, threadblock_bs, DY, DY * N * bs, dimX);
        cudaDeviceSynchronize();
      }
      cudaEventRecord(fft_end);
      cudaEventSynchronize(fft_begin);
      cudaEventSynchronize(fft_end);
      cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

      elapsed_time = elapsed_time / ntest;
      // printf("bs=%d dimX=%d dimY=%d M=%d, N=%d K=%d\n", bs, dimX, dimY, M, N, K);
      // printf("FFT_len=%d FFT_bs=%d\n", FFT_len, FFT_bs);
      printf("2D_D, bs=%-4d, DX=%-4d, DY=%-4d, N=%-4d, K=%-4d, TIME=%8.3fms\n",
        bs, DX, DY, N, K, elapsed_time);
    }
  // }}}}}
    return 0;
}