
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "cgemm.cuh"

using DataT = float2;
int main(int argc, char** argv){
    DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref;
    int M, N, K;
    freopen("input.txt", "r", stdin);
    scanf("%d%d%d", &M, &N, &K);
    long long int A_size = ((M + 127) / 128) * 128 * ((K + 127) / 128) * 128;
    long long int B_size = ((N + 127) / 128) * 128 * ((K + 127) / 128) * 128;
    long long int C_size = ((M + 127) / 128) * 128 * ((N + 127) / 128) * 128;
    A = (DataT*)malloc(sizeof(DataT) * A_size);
    B = (DataT*)malloc(sizeof(DataT) * B_size);
    C = (DataT*)malloc(sizeof(DataT) * C_size);
    C_ref = (DataT*)malloc(sizeof(DataT) * C_size);

    cudaMalloc((void**)&dA, sizeof(DataT) * A_size);
    cudaMalloc((void**)&dB, sizeof(DataT) * B_size);
    cudaMalloc((void**)&dC, sizeof(DataT) * C_size);
    cudaMalloc((void**)&dC_ref, sizeof(DataT) * C_size);

    for(long long int i = 0; i < A_size; ++i) {
      A[i].x = float(rand() % 5) + (rand() % 5) * 0.01;
      A[i].y = float(rand() % 5) + (rand() % 5) * 0.01;
    }
    for(long long int i = 0; i < B_size; ++i){
      B[i].x = float(rand() % 5) + (rand() % 5) * 0.01;
      B[i].y = float(rand() % 5) + (rand() % 5) * 0.01;
    }
    for(long long int i = 0; i < C_size; ++i){
      C[i].x = float(rand() % 5) + (rand() % 5) * 0.01;
      C[i].y = float(rand() % 5) + (rand() % 5) * 0.01;
    }

    cudaMemcpy(dA, A, sizeof(DataT) * A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);

    DataT alpha = {0.1,0.1} , beta = {0.1,0.1}; 

    int num_tests = 1;

    cgemm(M, N, K, dA, dB, dC, alpha, beta);
    
    cublasHandle_t handle;   
    cublasCreate(&handle);
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC, M);     
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    cudaDeviceSynchronize();
    for(int i = 0; i < 10; ++i){
      cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);
      cudaDeviceSynchronize();
    }  
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * 10 * double(M) * double(N) * double(K)) / (1e9)) / (elapsed / 1e3);
    printf("%d, %d, %d, %f, %f\n", M, N, K, elapsed, gflops);


    cudaMemcpy(C, dC, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, dC_ref, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);

    verify_vector((float*)C_ref, (float*)C, M * N * 2);

    return 0;
}