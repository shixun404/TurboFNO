#include <stdio.h>
#include <mma.h>
using namespace nvcuda;
#define warp_col_tiles 4
#define warp_row_tiles 4
#define SKEW_KERNEL_2 0
extern __shared__ float shared_mem[];
__global__ void zgemm_20(int M, int N, int K, float *A, float *B, float *C, float2 alpha, float2 beta){
    int threadblock_tile_M = 64, threadblock_tile_N = 128;
    int tid = threadIdx.x;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;
    int wid = tid / 32;
    float tmp_r, tmp_c, tmp_a, tmp_b;
    
    // __shared__ float shared_mem[((64 + SKEW_KERNEL_2) * 16 * 2) * 2];
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (((float2*)C));

    float2* sA = (shared_mem_float2 + (64 + SKEW_KERNEL_2) * 8 * 0);
    float2* sB = (shared_mem_float2 + (128 + SKEW_KERNEL_2) * 8 * 0 + (64 + SKEW_KERNEL_2) * 8 * 1);
    
    float2 mem_temp[8];
    
    float2 c[warp_col_tiles][warp_row_tiles][2];
    float2 a[2][4];
    float2 b[2][4];
    
    #pragma unroll
    for(int i = 0; i < warp_col_tiles; i++){
        #pragma unroll
        for(int j = 0; j < warp_row_tiles; j++){
            c[i][j][0].x = 0;
            c[i][j][0].y = 0;
            c[i][j][1].x = 0;
            c[i][j][1].y = 0;
            
        }
    }
    int offset = 0, offset_k = 0;
    int k = 0;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sA) + (0 + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&A[((bid_x * 64 + tid / 4) + (0 + tid % 4 + 0) * M) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sA) + (0 + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&A[((bid_x * 64 + tid / 4) + (0 + tid % 4 + 4) * M) * 2 + 0]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (0 + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((0 + tid % 4) + (bid_y * 128 + tid / 4) * K) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (0 + (tid % 4) + (tid / 4 + 128) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((0 + tid % 4 + 4) + (bid_y * 128 + tid / 4) * K) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (0 + (tid % 4) + (tid / 4 +  64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((0 + tid % 4) + (bid_y * 128 + tid / 4 + 64) * K) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (0 + (tid % 4) + (tid / 4 + 192) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((0 + tid % 4 + 4) + (bid_y * 128 + tid / 4 + 64) * K) * 2 + 0]));

    asm volatile("cp.async.commit_group;\n" ::);
    offset_k = (offset_k + 1) % 3;
    offset = (offset_k *  ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8));

    // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    //         : "l"(__cvta_generic_to_shared(sA) + (offset + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
    //             "l"(&A[((bid_x * 64 + tid / 4) + (8 + tid % 4 + 0) * M) * 2 + 0]));
    // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    //     : "l"(__cvta_generic_to_shared(sA) + (offset + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
    //         "l"(&A[((bid_x * 64 + tid / 4) + (8 + tid % 4 + 4) * M) * 2 + 0]));

    // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    //     : "l"(__cvta_generic_to_shared(sB) + (offset + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
    //         "l"(&B[((8 + tid % 4) + (bid_y * 64 + tid / 4) * K) * 2 + 0]));
    // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
    //     : "l"(__cvta_generic_to_shared(sB) + (offset + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
    //         "l"(&B[((8 + tid % 4 + 4) + (bid_y * 64 + tid / 4) * K) * 2 + 0]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sA) + (offset + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&A[((bid_x * 64 + tid / 4) + (8 + tid % 4 + 0) * M) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sA) + (offset + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&A[((bid_x * 64 + tid / 4) + (8 + tid % 4 + 4) * M) * 2 + 0]));

    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (offset + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((8 + tid % 4) + (bid_y * 128 + tid / 4) * K) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (offset + (tid % 4) + (tid / 4 + 128) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((8 + tid % 4 + 4) + (bid_y * 128 + tid / 4) * K) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (offset + (tid % 4) + (tid / 4 +  64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((8 + tid % 4) + (bid_y * 128 + tid / 4 + 64) * K) * 2 + 0]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(__cvta_generic_to_shared(sB) + (offset + (tid % 4) + (tid / 4 + 192) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&B[((8 + tid % 4 + 4) + (bid_y * 128 + tid / 4 + 64) * K) * 2 + 0]));


    int offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
    asm volatile("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 1;\n" ::);
    // asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
    b[0][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    b[0][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    b[0][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    b[0][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    
    

    a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));


    

    #pragma unroll
    for(k = 0; k < K - 16; k += 8){
        int offset_next_k = (offset_k + 1) % 3;
        int offset_next = (offset_next_k *  ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "l"(__cvta_generic_to_shared(sA) + (offset_next + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
        //         "l"(&A[((bid_x * 64 + tid / 4) + ((k + 16) + tid % 4 + 0) * M) * 2 + 0]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "l"(__cvta_generic_to_shared(sA) + (offset_next + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
        //         "l"(&A[((bid_x * 64 + tid / 4) + ((k + 16) + tid % 4 + 4) * M) * 2 + 0]));

        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "l"(__cvta_generic_to_shared(sB) + (offset_next + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
        //         "l"(&B[(((k + 16) + tid % 4) + (bid_y * 64 + tid / 4) * K) * 2 + 0]));
        // asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        //     : "l"(__cvta_generic_to_shared(sB) + (offset_next + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
        //         "l"(&B[(((k + 16) + tid % 4 + 4) + (bid_y * 64 + tid / 4) * K) * 2 + 0]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sA) + (offset_next + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
            "l"(&A[((bid_x * 64 + tid / 4) + ((k + 16) + tid % 4 + 0) * M) * 2 + 0]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sA) + (offset_next + (tid % 4) + (tid / 4 + 64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&A[((bid_x * 64 + tid / 4) + ((k + 16) + tid % 4 + 4) * M) * 2 + 0]));

        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sB) + (offset_next + (tid % 4) + (tid / 4 +  0) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&B[(((k + 16) + tid % 4) + (bid_y * 128 + tid / 4) * K) * 2 + 0]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sB) + (offset_next + (tid % 4) + (tid / 4 + 128) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&B[(((k + 16) + tid % 4 + 4) + (bid_y * 128 + tid / 4) * K) * 2 + 0]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sB) + (offset_next + (tid % 4) + (tid / 4 +  64) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&B[(((k + 16) + tid % 4) + (bid_y * 128 + tid / 4 + 64) * K) * 2 + 0]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(__cvta_generic_to_shared(sB) + (offset_next + (tid % 4) + (tid / 4 + 192) * (4 + SKEW_KERNEL_2)) * sizeof(float) * 2), 
                "l"(&B[(((k + 16) + tid % 4 + 4) + (bid_y * 128 + tid / 4 + 64) * K) * 2 + 0]));

        int offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
        b[1][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        b[1][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        b[1][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        b[1][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8 + 128) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        

        a[1][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        a[1][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        a[1][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        a[1][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8 + 64) * (4 + SKEW_KERNEL_2) + ((tid % 32) % 4));
        

        tmp_r = a[1][0].x, tmp_c = a[1][0].y;

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_r), "=d"(tmp_c)
        : "d"(a[1][2].x), "d"(a[1][2].y), "d"(a[1][1].x), "d"(a[1][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_r), "=d"(tmp_c)
        : "d"(a[1][2].x), "d"(a[1][2].y), "d"(a[1][3].x), "d"(a[1][3].y));

        tmp_a = b[1][0].x, tmp_b = b[1][0].y;
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_a), "=d"(tmp_b)
        : "d"(b[1][2].x), "d"(b[1][2].y), "d"(b[1][1].x), "d"(b[1][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_a), "=d"(tmp_b)
        : "d"(b[1][2].x), "d"(b[1][2].y), "d"(b[1][3].x), "d"(b[1][3].y));
        


        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_r), "d"(tmp_a), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_c), "d"(-tmp_b), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_r), "d"(tmp_b), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_c), "d"(tmp_a), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        
        #pragma unroll
        for(int kk = 0; kk < 2; ++kk){   
            #pragma unroll
            for(int i = 0; i < warp_col_tiles; ++i){
                #pragma unroll
                for(int j = 0; j < warp_row_tiles; ++j){
                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].x), "=d"(c[i][j][1].x)
                    : "d"(a[kk][j].x), "d"(b[kk][i].x), "d"(c[i][j][0].x), "d"(c[i][j][1].x));

                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].x), "=d"(c[i][j][1].x)
                    : "d"(a[kk][j].y), "d"(-b[kk][i].y), "d"(c[i][j][0].x), "d"(c[i][j][1].x));

                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].y), "=d"(c[i][j][1].y)
                    : "d"(a[kk][j].x), "d"(b[kk][i].y), "d"(c[i][j][0].y), "d"(c[i][j][1].y));

                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].y), "=d"(c[i][j][1].y)
                    : "d"(a[kk][j].y), "d"(b[kk][i].x), "d"(c[i][j][0].y), "d"(c[i][j][1].y));
                }
            }
        }
        offset_k = (offset_k + 1) % 3;
        offset = (offset_k * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8));
        offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
        // const int n_wait = (k != (K - 24));
        asm ("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        // asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();
        b[0][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[0][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[0][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[0][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        

        a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));

        tmp_r = a[0][0].x, tmp_c = a[0][0].y;

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_r), "=d"(tmp_c)
        : "d"(a[0][2].x), "d"(a[0][2].y), "d"(a[0][1].x), "d"(a[0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_r), "=d"(tmp_c)
        : "d"(a[0][2].x), "d"(a[0][2].y), "d"(a[0][3].x), "d"(a[0][3].y));

        tmp_a = b[0][0].x, tmp_b = b[0][0].y;
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_a), "=d"(tmp_b)
        : "d"(b[0][2].x), "d"(b[0][2].y), "d"(b[0][1].x), "d"(b[0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(tmp_a), "=d"(tmp_b)
        : "d"(b[0][2].x), "d"(b[0][2].y), "d"(b[0][3].x), "d"(b[0][3].y));
        


        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_r), "d"(tmp_a), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_c), "d"(-tmp_b), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_r), "d"(tmp_b), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
        : "=d"(c[0][0][0].y), "=d"(c[0][0][1].y)
        : "d"(tmp_c), "d"(tmp_a), "d"(c[0][0][0].y), "d"(c[0][0][1].y));
    }
    // asm ("cp.async.commit_group;\n" ::);
    // asm volatile("cp.async.wait_all;\n" ::);
    // __syncthreads();
    // b[0][0] = *(sB + offset_cur + ((wid % 4) * 16 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // b[0][1] = *(sB + offset_cur + ((wid % 4) * 16 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    

    // a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    // a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));


    for(; k < K; k += 8){
        int offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
        b[1][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[1][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[1][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[1][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8 + 128) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        

        a[1][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[1][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[1][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[1][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8 + 64) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        
        #pragma unroll
        for(int kk = 0; kk < 2; ++kk){   
            #pragma unroll
            for(int i = 0; i < warp_col_tiles; ++i){
                #pragma unroll
                for(int j = 0; j < warp_row_tiles; ++j){
                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].x), "=d"(c[i][j][1].x)
                    : "d"(a[kk][j].x), "d"(b[kk][i].x), "d"(c[i][j][0].x), "d"(c[i][j][1].x));

                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].x), "=d"(c[i][j][1].x)
                    : "d"(a[kk][j].y), "d"(-b[kk][i].y), "d"(c[i][j][0].x), "d"(c[i][j][1].x));

                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].y), "=d"(c[i][j][1].y)
                    : "d"(a[kk][j].x), "d"(b[kk][i].y), "d"(c[i][j][0].y), "d"(c[i][j][1].y));

                    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[i][j][0].y), "=d"(c[i][j][1].y)
                    : "d"(a[kk][j].y), "d"(b[kk][i].x), "d"(c[i][j][0].y), "d"(c[i][j][1].y));
                }
            }
        }
        offset_k = (offset_k + 1) % 3;
        offset = (offset_k * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8));
        offset_cur = ((offset_k + 2) % 3) * ((64 + SKEW_KERNEL_2 + 128 + SKEW_KERNEL_2) * 8);
        b[0][0] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[0][1] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[0][2] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        b[0][3] = *(sB + offset_cur + ((wid % 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    

        a[0][0] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 0 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[0][1] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 1 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[0][2] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 2 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
        a[0][3] = *(sA + offset_cur + ((wid / 4) * 32 + (tid % 32) / 4 + 3 * 8) * (4 + SKEW_KERNEL_2) + (0 * 4 + (tid % 32) % 4));
    }

    
    
    #pragma unroll
    for(int i = 0; i < warp_col_tiles; ++i){
        #pragma unroll
        for(int j = 0; j < warp_row_tiles; ++j){
            mem_temp[0] = *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 0) * M);
            mem_temp[1] = *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 1) * M);
            #pragma unroll
            for(int ii = 0; ii < 2; ii++) {
                mem_temp[ii + 2].x = alpha.x * c[i][j][ii].x + beta.x * mem_temp[ii].x - 
                                        alpha.y * c[i][j][ii].y - beta.y * mem_temp[ii].y;
                mem_temp[ii + 2].y = alpha.x * c[i][j][ii].y + beta.x * mem_temp[ii].y +
                                        alpha.y * c[i][j][ii].x + beta.y * mem_temp[ii].x;       
            }
            *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 0) * M) = mem_temp[0 + 2];
            *(gC + (bid_x * 64 + (wid / 4) * 32 + j * 8 + ((tid % 32) / 4)) + (bid_y * 128 + (wid % 4) * 32 + i * 8 + ((tid % 32) % 4) * 2 + 1) * M) = mem_temp[1 + 2];

        }
    }

}