
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

void test_cufft(float2* input_d, float2* output_d, 
  float2* output_cufft, long long int N, size_t bs, int ntest) {
  cufftHandle plan;
    float gflops, elapsed_time, mem_bandwidth;
    


    cufftCreate(&plan);

    cufftPlan1d(&plan, N, CUFFT_C2C, bs);


    cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(input_d), 
                        reinterpret_cast<cufftComplex*>(output_d), 
                        CUFFT_FORWARD);
        cudaDeviceSynchronize();
        cudaEvent_t fft_begin, fft_end;
        cudaEventCreate(&fft_begin);
    cudaEventCreate(&fft_end);

    cudaEventRecord(fft_begin);
    for (int i = 0; i < ntest; ++i){
        cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(input_d), 
                        reinterpret_cast<cufftComplex*>(output_d), 
                        CUFFT_FORWARD);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(fft_end);
    cudaEventSynchronize(fft_begin);
    cudaEventSynchronize(fft_end);
    cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

    elapsed_time = elapsed_time / ntest;
    gflops = 5 * N * log2f(N) * bs / elapsed_time * 1000 / 1000000000.f;
    
    mem_bandwidth = (float)(N * bs * 8 * 2) / (elapsed_time) * 1000.f / 1000000000.f;
    printf("cuFFT, N=%d, BS=%d, TIME=%8.3f, GFLOPS%8.3f, MEM=%8.3f\n",  (int)log2f(N),  (int)log2f(bs), elapsed_time, gflops, mem_bandwidth);

    cudaMemcpy(output_cufft, output_d, N * bs * sizeof(float2), 
                   cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
}

void test_myfft(float2* input_d, float2* output_d, 
  float2* output_host, long long int N, size_t bs, int threadblock_bs, int ntest) {
    float gflops, elapsed_time, mem_bandwidth;
    dim3 gridDim(bs / threadblock_bs, 1, 1);
    dim3 blockDim(N / thread_bs[int(log2f(N)) - 7] * threadblock_bs, 1, 1);
    int shmem_size = sizeof(float2) * N * threadblock_bs;
    printf("%d %d %d\n", gridDim.x, blockDim.x, shmem_size);
    void (*myfft_ptr)(float2 *, float2 *, int) = turboFFTArr[int(log2f(N)) - 7];
    myfft_ptr<<<gridDim, blockDim, shmem_size>>>(input_d, output_d, threadblock_bs);
    cudaDeviceSynchronize();
    cudaEvent_t fft_begin, fft_end;

    cudaEventCreate(&fft_begin);
    cudaEventCreate(&fft_end);

    cudaEventRecord(fft_begin);
    // fft_7<<<gridDim, blockDim, shmem_size>>>(input_d, output_d, threadblock_bs);
    // cudaError_t err;  // 获取最近的错误
    

    for (int i = 0; i < ntest; ++i){
        myfft_ptr<<<gridDim, blockDim, shmem_size>>>(input_d, output_d, threadblock_bs);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(fft_end);
    cudaEventSynchronize(fft_begin);
    cudaEventSynchronize(fft_end);
    cudaEventElapsedTime(&elapsed_time, fft_begin, fft_end);

    elapsed_time = elapsed_time / ntest;
    gflops = 5 * N * log2f(N) * bs / elapsed_time * 1000 / 1000000000.f;
    
    // printf("cuFFT finished: T=%8.3fms, FLOPS=%8.3fGFLOPS\n", elapsed_time, gflops);
    mem_bandwidth = (float)(N * bs * 8 * 2) / (elapsed_time) * 1000.f / 1000000000.f;
    printf("TurboFFT, N=%d, BS=%d, TIME=%8.3f, GFLOPS%8.3f, MEM=%8.3f\n",  (int)log2f(N),  (int)log2f(bs), elapsed_time, gflops, mem_bandwidth);

    checkCudaErrors(cudaMemcpy(output_host, output_d, N * bs * sizeof(float2), 
                   cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv){
    if(argc < 6){
      printf("Usage: %s 0 M N K ntest\n", argv[0]);
      printf("Usage: %s 1 N BS threadblock_bs ntest\n", argv[0]);
      return 1;
    }
    if(argv[1] == 0){
      DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref;
    int M, N, K;
      M = atoi(argv[2]);
      N = atoi(argv[3]);
      K = atoi(argv[4]);
      int num_tests = argc > 5 ? atoi(argv[5]) : 1;
      // freopen("input.txt", "r", stdin);
      // scanf("%d%d%d", &M, &N, &K);
      long long int A_size = M * K;
      long long int B_size = N * K;
      long long int C_size = M * N;
      A = (DataT*)malloc(sizeof(DataT) * A_size);
      B = (DataT*)malloc(sizeof(DataT) * B_size);
      C = (DataT*)malloc(sizeof(DataT) * C_size);
      C_ref = (DataT*)malloc(sizeof(DataT) * C_size);

      cudaMalloc((void**)&dA, sizeof(DataT) * A_size);
      cudaMalloc((void**)&dB, sizeof(DataT) * B_size);
      cudaMalloc((void**)&dC, sizeof(DataT) * C_size);
      cudaMalloc((void**)&dC_ref, sizeof(DataT) * C_size);

      generate_random_vector((float*)A, M * K * 2);
      generate_random_vector((float*)B, N * K * 2);
      fill_vector((float*)C, 0, M * N * 2);

      cudaMemcpy(dA, A, sizeof(DataT) * A_size, cudaMemcpyHostToDevice);
      cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice);
      cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);
      cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);

      DataT alpha = {1.0, -1.0} , beta = {-1.0, 1.0}; 
      // DataT alpha = {1.0, 0} , beta = {1.0, 0}; 

      
      dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M, (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
      dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1); 
      int shmem_size = sizeof(DataT) * (THREADBLOCK_M * THREADBLOCK_K + THREADBLOCK_N * THREADBLOCK_K) * 2;  

      cublasHandle_t handle;   
      cublasCreate(&handle);
      cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);     
      cudaEvent_t beg, end;
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      float elapsed;
      cudaDeviceSynchronize();
      cudaEventRecord(beg);
      cudaDeviceSynchronize();
      for(int i = 0; i < num_tests; ++i){
        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);
        cudaDeviceSynchronize();
      }  
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed, beg, end);
      double gflops = (double(2 * num_tests * double(M) * double(N) * double(K)) / (1e9)) / (elapsed / 1e3);
      printf("cublas: %d, %d, %d, %f, %f\n", M, N, K, elapsed, gflops);


          
      {
        fused_fft_cgemm<<<gridDim, blockDim, shmem_size>>>(M, N, K, dA, dB, dC, alpha, beta);
        cudaEvent_t beg, end;
        cudaEventCreate(&beg);
        cudaEventCreate(&end);
        float elapsed;
        cudaDeviceSynchronize();
        cudaEventRecord(beg);
        cudaDeviceSynchronize();
        for(int i = 0; i < num_tests; ++i){
          fused_fft_cgemm<<<gridDim, blockDim, shmem_size>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }  
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, beg, end);
        double gflops = (double(2 * num_tests * double(M) * double(N) * double(K)) / (1e9)) / (elapsed / 1e3);
        printf("cgemm: %d, %d, %d, %f, %f\n", M, N, K, elapsed, gflops);
    }

      cudaMemcpy(C, dC, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);

      verify_vector((float*)C_ref, (float*)C, M * N * 2);
  }
  else{
    DataT *input, *input_d, *output, *output_ref, *output_d, *output_ref_d;
    int N = atoi(argv[2]);
    int BS = atoi(argv[3]);
    int threadblock_bs = atoi(argv[4]);
    int ntest = atoi(argv[5]);

    long long int input_size = N * BS;
    long long int output_size = N * BS;
    input = (DataT*)malloc(sizeof(DataT) * input_size);
    output = (DataT*)malloc(sizeof(DataT) * output_size);
    output_ref = (DataT*)malloc(sizeof(DataT) * output_size);

    cudaMalloc((void**)&input_d, sizeof(DataT) * input_size);
    cudaMalloc((void**)&output_d, sizeof(DataT) * output_size);
    cudaMalloc((void**)&output_ref_d, sizeof(DataT) * output_size);

    generate_random_vector((float*)input, input_size * 2);
    // fill_vector((float*)input, 0, output_size * 2);
    // input[0].x = 1.0;
    // input[0].y = 0.0;

    cudaMemcpy(input_d, input, sizeof(DataT) * input_size, cudaMemcpyHostToDevice);

    test_cufft(input_d, output_ref_d, output_ref, N, BS, ntest);
    test_myfft(input_d, output_d, output, N, BS, threadblock_bs, ntest);

    verify_vector((float*)output_ref, (float*)output, output_size * 2);

  }

    return 0;
}