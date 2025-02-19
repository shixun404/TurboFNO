#include <stdio.h>
#include <mma.h>
#define threadblock_M 64
#define threadblock_N 64
#define threadblock_K 8
#define warp_M 32
#define warp_N 16
#define thread_M 4
#define thread_N 4
#define thread_num blockDim.x
#define tid threadIdx.x
#define bid_x blockIdx.x
#define bid_y blockIdx.y
#define load_per_thread_A threadblock_M * threadblock_K / thread_num
#define load_per_thread_B threadblock_N * threadblock_K / thread_num
extern __shared__ float shared_mem[];
__global__ void cgemm(int M, int N, int K, float *A, float *B, float *C, float2 alpha, float2 beta){
    int tid = threadIdx.x;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;
    int wid = tid / 32;
    float tmp_r, tmp_c, tmp_a, tmp_b;
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
    float2 tmp_b[load_per_thread_b];
    
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
    for(int i = 0; i < load_per_thread_A; i++){
        tmp_B[i] = gA[bid_x * threadblock_M + (tid * load_per_thread_A + i) % threadblock_M
        + (tid * load_per_thread_A + i) / threadblock_M * M];
    }
    
    // __syncthreads();
    // b[0][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // b[0][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // b[0][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // b[0][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    
    

    // a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));


    

    // #pragma unroll
    // for(k = 0; k < K - 16; k += 8){
    //     int offset_next_k = (offset_k + 1) % 3;
    //     int offset_next = (offset_next_k *  ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8));
       
    //     int offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
    //     b[1][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
    //     b[1][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
    //     b[1][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
    //     b[1][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        

    //     a[1][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
    //     a[1][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
    //     a[1][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
    //     a[1][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        

    //     tmp_r = a[1][0].x, tmp_c = a[1][0].y;

    //     #pragma unroll
    //     for(int kk = 0; kk < 2; ++kk){   
    //         #pragma unroll
    //         for(int i = 0; i < warp_col_tiles; ++i){
    //             #pragma unroll
    //             for(int j = 0; j < warp_row_tiles; ++j){
    //             }
    //         }
    //     }
        
    //     __syncthreads();
    //     b[0][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[0][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[0][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[0][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        

    //     a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));

    //     tmp_r = a[0][0].x, tmp_c = a[0][0].y;

    // }
   
    // for(; k < K; k += 8){
    //     int offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
    //     b[1][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[1][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[1][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[1][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        

    //     a[1][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[1][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[1][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[1][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        
    //     #pragma unroll
    //     for(int kk = 0; kk < 2; ++kk){   
    //         #pragma unroll
    //         for(int i = 0; i < warp_col_tiles; ++i){
    //             #pragma unroll
    //             for(int j = 0; j < warp_row_tiles; ++j){
                  
    //             }
    //         }
    //     }
       
    //     b[0][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[0][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[0][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     b[0][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    

    //     a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    //     a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // }

    
    
    // #pragma unroll
    // for(int i = 0; i < warp_col_tiles; ++i){
    //     #pragma unroll
    //     for(int j = 0; j < warp_row_tiles; ++j){
    //         mem_temp[0] = *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 0) * M);
    //         mem_temp[1] = *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 1) * M);
    //         #pragma unroll
    //         for(int ii = 0; ii < 2; ii++) {
    //             mem_temp[ii + 2].x = alpha.x * c[i][j][ii].x + beta.x * mem_temp[ii].x - 
    //                                     alpha.y * c[i][j][ii].y - beta.y * mem_temp[ii].y;
    //             mem_temp[ii + 2].y = alpha.x * c[i][j][ii].y + beta.x * mem_temp[ii].y +
    //                                     alpha.y * c[i][j][ii].x + beta.y * mem_temp[ii].x;       
    //         }
    //         *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 0) * M) = mem_temp[0 + 2];
    //         *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 1) * M) = mem_temp[1 + 2];

    //     }
    // }

}