#include <stdio.h>
#include <mma.h>
#define THREADBLOCK_M 64
#define THREADBLOCK_N 64
#define THREADBLOCK_K 8
#define WARP_M 32
#define WARP_N 16
#define THREAD_M 4
#define THREAD_N 4
#define WARP_NUM_ROW thredblock_M / WARP_M
#define THREAD_NUM_ROW WARP_M / THREAD_M
#define THREAD_NUM blockDim.x
#define TID threadIdx.x
#define WID threadIdx.x / 32
#define BID_X blockIdx.x
#define BID_Y blockIdx.y
#define LOAD_PER_THREAD_A THREADBLOCK_M * THREADBLOCK_K / THREAD_NUM
#define LOAD_PER_THREAD_B THREADBLOCK_N * THREADBLOCK_K / THREAD_NUM
extern __shared__ float shared_mem[];
__global__ void cgemm(int M, int N, int K, float2 *A, float2 *B, float2 *C, float2 alpha, float2 beta){
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (float2*)C;
    float2 * gA = (float2*)A;
    float2 * gB = (float2*)B;

    float2* sA = shared_mem_float2;
    float2* sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K;
    
    float2 mem_temp[8];
    
    float2 c[THREAD_M][THREAD_N];
    float2 a[2][THREAD_M];
    float2 b[2][THREAD_N];

    float2 tmp_A[LOAD_PER_THREAD_A];
    float2 tmp_B[LOAD_PER_THREAD_B];
    
    #pragma unroll
    for(int i = 0; i < THREAD_M; i++){
        #pragma unroll
        for(int j = 0; j < THREAD_N; j++){
            c[i][j].x = 0;
            c[i][j].y = 0;
            c[i][j].x = 0;
            c[i][j].y = 0;
            
        }
    }
    int offset = 0, offset_k = 0;
    int k = 0;
    
    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_A; i++){
        tmp_A[i] = gA[BID_X * THREADBLOCK_M + (TID * LOAD_PER_THREAD_A + i) % THREADBLOCK_M
        + (TID * LOAD_PER_THREAD_A + i) / THREADBLOCK_M * M];
    }

    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_B; i++){
        tmp_B[i] = gB[(TID * LOAD_PER_THREAD_B + i) / THREADBLOCK_K
        + (BID_Y * THREADBLOCK_N + (TID * LOAD_PER_THREAD_B + i) % THREADBLOCK_K) * K];
    }
    
    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_A; i++){
        sA[TID * LOAD_PER_THREAD_A + i] = tmp_A[i];
    }

    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_B; i++){
        sB[TID * LOAD_PER_THREAD_B + i] = tmp_B[i];
    }

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < THREAD_M; i++){
        a[0][i] = sA[(WID % warp_num_M) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i];
    }   
    #pragma unroll
    for(int i = 0; i < THREAD_N; i++){
        b[0][i] = sB[(WID / warp_num_M) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + i];
    }
    thread_prefetch = 0;
    #pragma unroll
    for(int k = 0; k < K; k += THREADBLOCK_K){


        #pragma unroll
        for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
            #pragma unroll
            for(int i = 0; i < THREAD_M; i++){
                a[thread_prefetch][i] = sA[thread_k * THREADBLOCK_M + (WID % warp_num_M) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i];
            }   
            #pragma unroll
            for(int i = 0; i < THREAD_N; i++){
                b[thread_prefetch][i] = sB[thread_k * THREADBLOCK_N + (WID / warp_num_M) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + i];
            }
            #pragma unroll
            for(int i = 0; i < THREAD_M; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_N; j++){
                    c[i][j].x += a[thread_prefetch][i].x * b[thread_prefetch][j].x - a[thread_prefetch][i].y * b[thread_prefetch][j].y;
                    c[i][j].y += a[thread_prefetch][i].x * b[thread_prefetch][j].y + a[thread_prefetch][i].y * b[thread_prefetch][j].x;
                }
            }
        }

    }


}