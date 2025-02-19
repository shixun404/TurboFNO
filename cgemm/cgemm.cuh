#include <stdio.h>
#include <mma.h>
#define threadblock_M 64
#define threadblock_N 64
#define threadblock_K 8
#define warp_M 32
#define warp_N 16
#define thread_M 4
#define thread_N 4
#define warp_num_row thredblock_M / warp_M
#define thread_num_row warp_M / thread_M
#define thread_num blockDim.x
#define tid threadIdx.x
#define wid threadIdx.x / 32
#define bid_x blockIdx.x
#define bid_y blockIdx.y
#define load_per_thread_A threadblock_M * threadblock_K / thread_num
#define load_per_thread_B threadblock_N * threadblock_K / thread_num
extern __shared__ float shared_mem[];
__global__ void cgemm(int M, int N, int K, float2 *A, float2 *B, float2 *C, float2 alpha, float2 beta){
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (float2*)C;
    float2 * gA = (float2*)A;
    float2 * gB = (float2*)B;

    float2* sA = shared_mem_float2;
    float2* sB = shared_mem_float2 + threadblock_M * threadblock_K;
    
    float2 mem_temp[8];
    
    float2 c[thread_M][thread_N];
    float2 a[2][thread_M];
    float2 b[2][thread_N];

    float2 tmp_A[load_per_thread_A];
    float2 tmp_B[load_per_thread_B];
    
    #pragma unroll
    for(int i = 0; i < thread_M; i++){
        #pragma unroll
        for(int j = 0; j < thread_N; j++){
            c[i][j].x = 0;
            c[i][j].y = 0;
            c[i][j].x = 0;
            c[i][j].y = 0;
            
        }
    }
    int offset = 0, offset_k = 0;
    int k = 0;
    
    #pragma unroll
    for(int i = 0; i < load_per_thread_A; i++){
        tmp_A[i] = gA[bid_x * threadblock_M + (tid * load_per_thread_A + i) % threadblock_M
        + (tid * load_per_thread_A + i) / threadblock_M * M];
    }

    #pragma unroll
    for(int i = 0; i < load_per_thread_B; i++){
        tmp_B[i] = gB[(tid * load_per_thread_B + i) / threadblock_K
        + (bid_y * threadblock_N + (tid * load_per_thread_B + i) % threadblock_K) * K];
    }
    
    #pragma unroll
    for(int i = 0; i < load_per_thread_A; i++){
        sA[tid * load_per_thread_A + i] = tmp_A[i];
    }

    #pragma unroll
    for(int i = 0; i < load_per_thread_B; i++){
        sB[tid * load_per_thread_B + i] = tmp_B[i];
    }

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < thread_M; i++){
        a[0][i] = sA[(wid % warp_num_M) * warp_M + ((tid % 32) %  thread_num_row) * thread_M + i];
    }   
    #pragma unroll
    for(int i = 0; i < thread_N; i++){
        b[0][i] = sB[(wid / warp_num_M) * warp_N + ((tid % 32) /  thread_num_row) * thread_N + i];
    }

    #pragma unroll
    for(int k = 0; k < K; k += threadblock_K){

        #pragma unroll
        for(int i = 0; i < thread_M; i++){
            
            #pragma unroll
            for(int j = 0; j < thread_N; j++){
                c[i][j].x += a[thread_prefetch][i].x * b[thread_prefetch][j].x - a[thread_prefetch][i].y * b[thread_prefetch][j].y;
                c[i][j].y += a[thread_prefetch][i].x * b[thread_prefetch][j].y + a[thread_prefetch][i].y * b[thread_prefetch][j].x;
            }

        }

    }


}