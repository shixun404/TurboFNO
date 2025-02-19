#include <stdio.h>
#include <cublas_v2.h>  
#include "utils/utils.cuh"
#include "cgemm.cuh"       
#define PPP 1    
#include <cuda_runtime.h> 
#include <helper_functions.h>           
#include <helper_cuda.h>  
#define BM 64
#define BN 64
#define BK 8

int main(int argc, char **argv)
{        
    if (argc < 2) {  
        printf("Please select a kernel (range 0 - 1, here 0 is for NVIDIA cuBLAS).\n");
         exit(-1);
    }
    srand(10);  
    int kernel_number = atoi(argv[1]);
    int num_tests = 10;
    // const int NSPLIT = atoi(argv[4]);
    int start_size = atoi(argv[2]);   
    int end_size = atoi(argv[3]); 
    int gap_size = 256;
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
        printf("%8.2d|", max_size);
    }
    printf("\n");  
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
        printf("%8.2f|", min(8.1, float(max_size) * 31.25 / 1e3));
    } 

    printf("\n");
    // int threads_x = atoi(argv[4]); 
    float2 alpha, beta; 
    alpha.x = 1.0, alpha.y = 1.0;
	beta.x = 1.0, beta.y = 1.0;  
    int max_size = end_size;
    float *A = NULL, *B = NULL, *C_ref = NULL, *C = NULL;
    float *dA = NULL,*dB = NULL, *dC_ref = NULL, *dC = NULL;
    int size = max_size * sizeof (int);
    int deviceId;
    cudaGetDevice(&deviceId); 
    cudaDeviceProp props = getDetails(deviceId);
    
    A = (float *)malloc(sizeof(float) * max_size * max_size * 2);
    B = (float *)malloc(sizeof(float) * max_size * max_size * 2);
    C = (float *)malloc(sizeof(float) * max_size * max_size * 2);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size * 2);
    
    generate_random_matrix_float(A, max_size);
    generate_random_matrix_float(A + max_size * max_size, max_size);
     
    generate_random_matrix_float(B, max_size);
    generate_random_matrix_float(B + max_size * max_size, max_size);
    
    generate_random_matrix_float(C, max_size);
    generate_random_matrix_float(C + max_size * max_size, max_size);
    
    // for(int i = 0; i < max_size; ++i){
    //     for(int j = 0; j < max_size; ++j){  
    //         C[j + i * max_size] = (float)(j + i * max_size);
    //         C[j + i * max_size + max_size * max_size] = (float)(j + i * max_size);
    //         A[j * 2 + i * max_size * 2] = (float)j;
    //         A[j * 2 + i * max_size * 2 + 1] = (float)i;
    //         B[j * 2 + i * max_size * 2] = (float)j;
    //         B[j * 2 + i * max_size * 2 + 1] = (float)i;
 
    //     }
    // }
    copy_matrix_float(C, C_ref, max_size); 
    copy_matrix_float(C + max_size * max_size, C_ref + max_size * max_size , max_size);
   
    CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(float) * max_size * max_size * 2));
    CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(float) * max_size * max_size * 2));
    CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(float) * max_size * max_size * 2));
    CUDA_CALLER(cudaMalloc((void**) &dC_ref, sizeof(float) * max_size * max_size * 2));
      
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size * 2, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size * 2, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size * 2, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size * 2, cudaMemcpyHostToDevice));

    cublasHandle_t handle;   
    cublasCreate(&handle);      
      
    // if (!verify_matrix_float(C_ref, C, max_size) ||  
    //     !verify_matrix_float(C_ref + max_size * max_size, C + max_size * max_size, max_size)) {
    //     printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
    //     exit(-3);  
    // }    
    int M, N, K;        
    int shared_mem_size = ((BM + SKEW_KERNEL_2) * BK * 2 + (BN + SKEW_KERNEL_2) * BK * 2 ) * 2 * 8; 
    shared_mem_size = ((64 + 4) * 16 * 2) * 3 * 8;
    int shared_mem_size_18 = ((64 + 0) * 16 * 2) * 3 * 8;
    int shared_mem_size_64x128x8 = ((64 + 128) * 16) * 3 * 8;
    // cudaFuncSetAttribute(zgemm_20, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size_64x128x8);
                            
    // cudaFuncSetAttribute(zgemm_9, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
        M = max_size, N = max_size, K = max_size;
        cublasCgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, M, N, K, &alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, &beta, (cuFloatComplex*)dC_ref, M);
        
        if(kernel_number == 0)
        {   
            cublasCgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, M, N, K, &alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, &beta, (cuFloatComplex*)dC, M);
        } 
        // else if(kernel_number == 20){  
        //     dim3 blockDim(256);     
        //     dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 128));
        //     zgemm_20 <<<gridDim, blockDim, shared_mem_size_64x128x8>>>(M, N, K, dA, dB, dC, alpha, beta); 
        // }   
  
         
        cudaDeviceSynchronize();       
        cudaMemcpy(C, dC, sizeof(float) * max_size * max_size * 2, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();  
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * max_size * max_size * 2, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();         
    
        // if (!verify_matrix_float2(C_ref, C, max_size)) {
        //     printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
        // exit(-3);            
        // }  
        cudaEvent_t beg, end;   
        cudaEventCreate(&beg); 
        cudaEventCreate(&end);   
        float elapsed = 0;         
           
        if (kernel_number == 0){
            cudaEventRecord(beg);
            for(int ii = 0; ii < num_tests; ++ii){
                    cudaDeviceSynchronize();
                    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, &beta, (cuFloatComplex*)dC, M);
                    cudaDeviceSynchronize();
            }  
            cudaEventRecord(end);
            cudaEventSynchronize(beg);
            cudaEventSynchronize(end);   
        } 
        // else if(kernel_number == 20){
        //     cudaEventRecord(beg); 
        //     dim3 blockDim(256); 
        //     dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 128));
        //     for(int ii = 0; ii < num_tests; ++ii){
        //         cudaDeviceSynchronize();
        //         zgemm_20<<<gridDim, blockDim, shared_mem_size_64x128x8>>>(M, N, K, dA, dB, dC, alpha, beta);
        //         cudaDeviceSynchronize();
        //     }    
        //     cudaEventRecord(end);
        //     cudaEventSynchronize(beg); 
        //     cudaEventSynchronize(end);
        // } 
 
        cudaEventElapsedTime(&elapsed, beg, end);
        
        float gflops = float(8 * num_tests * float(M) * float(N) * float(K)) / (1e9);
        float perf = gflops / (elapsed / 1e3);
        printf("%8.2f,", perf);        
        fflush(stdout);
    }
    printf("\n");
}
