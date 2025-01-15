#include <stdio.h>
#include <cublas_v2.h>  
#include "utils/utils.cuh"
#include "kernels.cuh"       
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
    const int NSPLIT = atoi(argv[4]);
    int start_size = atoi(argv[2]);   
    int end_size = atoi(argv[3]); 
    int gap_size = 256;
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
        printf("%8.2d|", max_size);
    }
    printf("\n");  
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
        printf("%8.2f|", min(8.1, double(max_size) * 31.25 / 1e3));
    } 

    printf("\n");
    int threads_x = atoi(argv[4]); 
    double2 alpha, beta; 
    alpha.x = 1.0, alpha.y = 1.0;
	beta.x = 1.0, beta.y = 1.0;  
    int max_size = end_size;
    double *A = NULL, *B = NULL, *C_ref = NULL, *C = NULL;
    double *dA = NULL,*dB = NULL, *dC_ref = NULL, *dC = NULL;
    int size = max_size * sizeof (int);
    int deviceId;
    cudaGetDevice(&deviceId); 
    cudaDeviceProp props = getDetails(deviceId);
    
    A = (double *)malloc(sizeof(double) * max_size * max_size * 2);
    B = (double *)malloc(sizeof(double) * max_size * max_size * 2);
    C = (double *)malloc(sizeof(double) * max_size * max_size * 2);
    C_ref = (double *)malloc(sizeof(double) * max_size * max_size * 2);
    
    generate_random_matrix_double(A, max_size);
    generate_random_matrix_double(A + max_size * max_size, max_size);
     
    generate_random_matrix_double(B, max_size);
    generate_random_matrix_double(B + max_size * max_size, max_size);
    
    generate_random_matrix_double(C, max_size);
    generate_random_matrix_double(C + max_size * max_size, max_size);
    
    // for(int i = 0; i < max_size; ++i){
    //     for(int j = 0; j < max_size; ++j){  
    //         C[j + i * max_size] = (double)(j + i * max_size);
    //         C[j + i * max_size + max_size * max_size] = (double)(j + i * max_size);
    //         A[j * 2 + i * max_size * 2] = (double)j;
    //         A[j * 2 + i * max_size * 2 + 1] = (double)i;
    //         B[j * 2 + i * max_size * 2] = (double)j;
    //         B[j * 2 + i * max_size * 2 + 1] = (double)i;
 
    //     }
    // }
    copy_matrix_double(C, C_ref, max_size); 
    copy_matrix_double(C + max_size * max_size, C_ref + max_size * max_size , max_size);
   
    CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(double) * max_size * max_size * 2));
    CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(double) * max_size * max_size * 2));
    CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(double) * max_size * max_size * 2));
    CUDA_CALLER(cudaMalloc((void**) &dC_ref, sizeof(double) * max_size * max_size * 2));
      
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(double) * max_size * max_size * 2, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(double) * max_size * max_size * 2, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC, C, sizeof(double) * max_size * max_size * 2, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC_ref, C_ref, sizeof(double) * max_size * max_size * 2, cudaMemcpyHostToDevice));

    cublasHandle_t handle;   
    cublasCreate(&handle);      
      
    // if (!verify_matrix_double(C_ref, C, max_size) ||  
    //     !verify_matrix_double(C_ref + max_size * max_size, C + max_size * max_size, max_size)) {
    //     printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
    //     exit(-3);  
    // }    
    int M, N, K;        
    int shared_mem_size = ((BM + SKEW_KERNEL_2) * BK * 2 + (BN + SKEW_KERNEL_2) * BK * 2 ) * 2 * 8; 
    shared_mem_size = ((64 + 4) * 16 * 2) * 3 * 8;
    int shared_mem_size_18 = ((64 + 0) * 16 * 2) * 3 * 8;
    int shared_mem_size_64x128x8 = ((64 + 128) * 16) * 3 * 8;
    // shared_mem_size = 76800; 
    // cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
    // cudaFuncSetAttribute(zgemm_8, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    // cudaFuncSetAttribute(zgemm_12, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    // cudaFuncSetAttribute(zgemm_15, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    // cudaFuncSetAttribute(zgemm_16, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute(zgemm_17, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute(zgemm_18, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size_18);
    cudaFuncSetAttribute(zgemm_19, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute(zgemm_20, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size_64x128x8);
                            
    // cudaFuncSetAttribute(zgemm_9, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
        M = max_size, N = max_size, K = max_size;
        cublasZgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, M, N, K, &alpha, (cuDoubleComplex*)dA, M, (cuDoubleComplex*)dB, K, &beta, (cuDoubleComplex*)dC_ref, M);
        
        if(kernel_number == 0)
        {   
            cublasZgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, M, N, K, &alpha, (cuDoubleComplex*)dA, M, (cuDoubleComplex*)dB, K, &beta, (cuDoubleComplex*)dC, M);
        } 
        else if(kernel_number == 17){  
            dim3 blockDim(256);     
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 64));
            zgemm_17 <<<gridDim, blockDim, shared_mem_size>>>(M, N, K, dA, dB, dC, alpha, beta); 
        }    
        else if(kernel_number == 18){  
            dim3 blockDim(256);     
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 64));
            zgemm_18 <<<gridDim, blockDim, shared_mem_size_18>>>(M, N, K, dA, dB, dC, alpha, beta); 
        }   
        else if(kernel_number == 19){  
            dim3 blockDim(256);     
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 64));
            zgemm_19 <<<gridDim, blockDim, shared_mem_size>>>(M, N, K, dA, dB, dC, alpha, beta); 
        }   
        else if(kernel_number == 20){  
            dim3 blockDim(256);     
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 128));
            zgemm_20 <<<gridDim, blockDim, shared_mem_size_64x128x8>>>(M, N, K, dA, dB, dC, alpha, beta); 
        }   
  
         
        cudaDeviceSynchronize();       
        cudaMemcpy(C, dC, sizeof(double) * max_size * max_size * 2, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();  
        cudaMemcpy(C_ref, dC_ref, sizeof(double) * max_size * max_size * 2, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();         
    
        // if (!verify_matrix_double2(C_ref, C, max_size)) {
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
                    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, (cuDoubleComplex*)dA, M, (cuDoubleComplex*)dB, K, &beta, (cuDoubleComplex*)dC, M);
                    cudaDeviceSynchronize();
            }  
            cudaEventRecord(end);
            cudaEventSynchronize(beg);
            cudaEventSynchronize(end);   
        } 
        else if(kernel_number == 17){
            cudaEventRecord(beg); 
            dim3 blockDim(256); 
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 64));
            for(int ii = 0; ii < num_tests; ++ii){
                cudaDeviceSynchronize();
                zgemm_17<<<gridDim, blockDim, shared_mem_size>>>(M, N, K, dA, dB, dC, alpha, beta);
                cudaDeviceSynchronize();
            }
            cudaEventRecord(end);
            cudaEventSynchronize(beg); 
            cudaEventSynchronize(end);
        } 
        else if(kernel_number == 18){
            cudaEventRecord(beg);  
            dim3 blockDim(256); 
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 64));
            for(int ii = 0; ii < num_tests; ++ii){
                cudaDeviceSynchronize();
                zgemm_18<<<gridDim, blockDim, shared_mem_size_18>>>(M, N, K, dA, dB, dC, alpha, beta);
                cudaDeviceSynchronize();
            }
            cudaEventRecord(end);
            cudaEventSynchronize(beg); 
            cudaEventSynchronize(end);
        } 
        else if(kernel_number == 19){
            cudaEventRecord(beg); 
            dim3 blockDim(256); 
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 64));
            for(int ii = 0; ii < num_tests; ++ii){
                cudaDeviceSynchronize();
                zgemm_19<<<gridDim, blockDim, shared_mem_size>>>(M, N, K, dA, dB, dC, alpha, beta);
                cudaDeviceSynchronize();
            }
            cudaEventRecord(end);
            cudaEventSynchronize(beg); 
            cudaEventSynchronize(end);
        } 
        else if(kernel_number == 20){
            cudaEventRecord(beg); 
            dim3 blockDim(256); 
            dim3 gridDim(CEIL_DIV(max_size, 64), CEIL_DIV(max_size, 128));
            for(int ii = 0; ii < num_tests; ++ii){
                cudaDeviceSynchronize();
                zgemm_20<<<gridDim, blockDim, shared_mem_size_64x128x8>>>(M, N, K, dA, dB, dC, alpha, beta);
                cudaDeviceSynchronize();
            }    
            cudaEventRecord(end);
            cudaEventSynchronize(beg); 
            cudaEventSynchronize(end);
        } 
 
        cudaEventElapsedTime(&elapsed, beg, end);
        
        double gflops = double(8 * num_tests * double(M) * double(N) * double(K)) / (1e9);
        double perf = gflops / (elapsed / 1e3);
        printf("%8.2f,", perf);        
        fflush(stdout);
    }
    printf("\n");
}
