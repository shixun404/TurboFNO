#include <stdio.h>
#include <mma.h>

#define tab(a, b, c)            c.x = c.x + a.x * b.x; \
                                c.x = c.x - a.y * b.y; \
                                c.y = c.y + a.x * b.y; \
                                c.y = c.y + a.y * b.x;


extern __shared__ float shared_mem[];
__global__ __launch_bounds__(256) void cgemm(int M, int N, int K, float2 *A, float2 *B, float2 *C, float2 alpha, float2 beta){
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (float2*)C;
    float2 * gA = (float2*)A;
    float2 * gB = (float2*)B;

    float2* sA = shared_mem_float2;
    float2* sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K;

    float2 c[THREAD_M * THREAD_N];
    float2 c_load[THREAD_M * THREAD_N];
    float2 a[2 * THREAD_M];
    float2 b[2 * THREAD_N];

    float2 tmp_A[LOAD_PER_THREAD_A];
    float2 tmp_B[LOAD_PER_THREAD_B];
    
    // #pragma unroll
    // for(int i = 0; i < THREAD_M; i++){
    //     #pragma unroll
    //     for(int j = 0; j < THREAD_N; j++){
    //         c[i * THREAD_N + j].x = 0;
    //         c[i * THREAD_N + j].y = 0;
    //     }
    // }

    memset(c, 0, sizeof(c));

    
    int offset = 0, offset_k = 0;
    int k = 0;
    
    
    tmp_A[0] = gA[BID_Y * THREADBLOCK_M + (TID + 0 * blockDim.x) % (THREADBLOCK_M)
        + (TID + 0 * blockDim.x) / THREADBLOCK_M * M];
    tmp_A[1] = gA[BID_Y * THREADBLOCK_M + (TID + 1 * blockDim.x) % (THREADBLOCK_M)
        + (TID + 1 * blockDim.x) / THREADBLOCK_M * M];
    

    
    tmp_B[0] = gB[(TID + 0 * blockDim.x) % THREADBLOCK_K
        + (BID_X * THREADBLOCK_N + (TID + 0 * blockDim.x) / THREADBLOCK_K) * K];
    tmp_B[1] = gB[(TID + 1 * blockDim.x) % THREADBLOCK_K
        + (BID_X * THREADBLOCK_N + (TID + 1 * blockDim.x) / THREADBLOCK_K) * K];
    
    
    
    *(((float2*)sA) + TID + 0 * blockDim.x) = tmp_A[0];
    *(((float2*)sA) + TID + 1 * blockDim.x) = tmp_A[1];
    

    
    sB[(TID + 0 * blockDim.x) / THREADBLOCK_K + ((TID + 0 * blockDim.x) % THREADBLOCK_K) * THREADBLOCK_N] = ((float2*)tmp_B)[0];
    sB[(TID + 1 * blockDim.x) / THREADBLOCK_K + ((TID + 1 * blockDim.x) % THREADBLOCK_K) * THREADBLOCK_N] = ((float2*)tmp_B)[1];
    

    float4 tmp;
    #pragma unroll
    for(int j = 0; j < THREAD_N; j++){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i += 2){
            tmp = *((float4*)gC + BID_Y * THREADBLOCK_M / 2+ (WID / WARP_NUM_COL) * WARP_M / 2 + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M / 2 + i
            + (BID_X * THREADBLOCK_N + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + j) * M / 2);
            c_load[i * THREAD_N + j].x = tmp.x;
            c_load[i * THREAD_N + j].y = tmp.y;

            c_load[(i + 1) * THREAD_N + j].x = tmp.z;
            c_load[(i + 1) * THREAD_N + j].y = tmp.w;
        }
    }


    __syncthreads();

    
    a[0] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 0];
    a[1] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 1];
    a[2] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 2];
    a[3] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 3];
    
    
    b[0] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 0];
    b[1] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 1];
    b[2] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 2];
    b[3] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 3];
    
    int thread_prefetch = 0;
    int warp_prefetch = 1;
    // Main Loop along K
    #pragma unroll
    for(int k = 0; k < K - THREADBLOCK_K; k += THREADBLOCK_K){

        // Prefetech from global memory
        
        tmp_A[0] = gA[BID_Y * THREADBLOCK_M + (TID + 0 * blockDim.x) % THREADBLOCK_M
                + (TID + 0 * blockDim.x) / (THREADBLOCK_M) * (M) + (k + THREADBLOCK_K) * (M)];
        tmp_A[1] = gA[BID_Y * THREADBLOCK_M + (TID + 1 * blockDim.x) % THREADBLOCK_M
                + (TID + 1 * blockDim.x) / (THREADBLOCK_M) * (M) + (k + THREADBLOCK_K) * (M)];
        
    
        
        tmp_B[0] = gB[(TID + 0 * blockDim.x) % THREADBLOCK_K + (k + THREADBLOCK_K)
                + (BID_X * THREADBLOCK_N + (TID + 0 * blockDim.x) / THREADBLOCK_K) * K];
        tmp_B[1] = gB[(TID + 1 * blockDim.x) % THREADBLOCK_K + (k + THREADBLOCK_K)
                + (BID_X * THREADBLOCK_N + (TID + 1 * blockDim.x) / THREADBLOCK_K) * K];
        

        // Thread-level GEMM
        #pragma unroll
        for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
            
            a[((thread_prefetch + 1) % 2) * THREAD_M + 0] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 0];
            a[((thread_prefetch + 1) % 2) * THREAD_M + 1] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 1];
            a[((thread_prefetch + 1) % 2) * THREAD_M + 2] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 2];
            a[((thread_prefetch + 1) % 2) * THREAD_M + 3] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 3];
               
            
            b[((thread_prefetch + 1) % 2) * THREAD_N + 0] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 0];
            b[((thread_prefetch + 1) % 2) * THREAD_N + 1] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 1];
            b[((thread_prefetch + 1) % 2) * THREAD_N + 2] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 2];
            b[((thread_prefetch + 1) % 2) * THREAD_N + 3] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 3];
            
            tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 0], c[0 * THREAD_N + 0]);

            tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 1], c[0 * THREAD_N + 1]);

            tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 2], c[0 * THREAD_N + 2]);
            
            tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 3], c[0 * THREAD_N + 3]);
            
            tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 0], c[1 * THREAD_N + 0]);

            tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 1], c[1 * THREAD_N + 1]);

            tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 2], c[1 * THREAD_N + 2]);

            tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 3], c[1 * THREAD_N + 3]);

            tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 0], c[2 * THREAD_N + 0]);
            
            tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 1], c[2 * THREAD_N + 1]);

            tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 2], c[2 * THREAD_N + 2]);

            tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 3], c[2 * THREAD_N + 3]);

            tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 0], c[3 * THREAD_N + 0]);

            tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 1], c[3 * THREAD_N + 1]);

            tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 2], c[3 * THREAD_N + 2]);

            tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 3], c[3 * THREAD_N + 3]);

            

    
            thread_prefetch = (thread_prefetch + 1) % 2;
        }
        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 0], c[0 * THREAD_N + 0]);

        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 1], c[0 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 2], c[0 * THREAD_N + 2]);
        
        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 3], c[0 * THREAD_N + 3]);
        
        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 0], c[1 * THREAD_N + 0]);

        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 1], c[1 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 2], c[1 * THREAD_N + 2]);

        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 3], c[1 * THREAD_N + 3]);

        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 0], c[2 * THREAD_N + 0]);
        
        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 1], c[2 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 2], c[2 * THREAD_N + 2]);

        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 3], c[2 * THREAD_N + 3]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 0], c[3 * THREAD_N + 0]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 1], c[3 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 2], c[3 * THREAD_N + 2]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 3], c[3 * THREAD_N + 3]);

        // Store prefeteched global data to shared
        
        *(((float2*)shared_mem_float2) + (warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K) + TID + 0 * blockDim.x) = tmp_A[0];
        *(((float2*)shared_mem_float2) + (warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K) + TID + 1 * blockDim.x) = tmp_A[1];
        
    
        
        shared_mem_float2[THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N + 2) * THREADBLOCK_K
                                 + (TID + 0 * blockDim.x) / THREADBLOCK_K + ((TID + 0 * blockDim.x) % THREADBLOCK_K) + (((TID + 0 * blockDim.x) % THREADBLOCK_K)) * THREADBLOCK_N] = ((float2*)tmp_B)[0];
        shared_mem_float2[THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N + 2) * THREADBLOCK_K
                                 + (TID + 1 * blockDim.x) / THREADBLOCK_K + ((TID + 1 * blockDim.x) % THREADBLOCK_K) + (((TID + 1 * blockDim.x) % THREADBLOCK_K)) * THREADBLOCK_N] = ((float2*)tmp_B)[1];
        
        
        __syncthreads();
        
        // Prefetech from shared memory
        sA = shared_mem_float2 + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N + 2) * THREADBLOCK_K;
        sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N + 2) * THREADBLOCK_K;
        thread_prefetch = 0;
    
        a[0] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 0];
        a[1] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 1];
        a[2] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 2];
        a[3] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 3];
        
        
        b[0] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 0];
        b[1] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 1];
        b[2] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 2];
        b[3] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * 2 + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 3];
        warp_prefetch = (warp_prefetch + 1) % 2;
    }
    // Thread-level GEMM
    #pragma unroll
    for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
        
        a[((thread_prefetch + 1) % 2) * THREAD_M + 0] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 0];
        a[((thread_prefetch + 1) % 2) * THREAD_M + 1] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 1];
        a[((thread_prefetch + 1) % 2) * THREAD_M + 2] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 2];
        a[((thread_prefetch + 1) % 2) * THREAD_M + 3] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + 3];
           
        
        b[((thread_prefetch + 1) % 2) * THREAD_N + 0] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 0];
        b[((thread_prefetch + 1) % 2) * THREAD_N + 1] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 1];
        b[((thread_prefetch + 1) % 2) * THREAD_N + 2] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 2];
        b[((thread_prefetch + 1) % 2) * THREAD_N + 3] = sB[(thread_k + 1) * (THREADBLOCK_N + 2) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + 3];
        
        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 0], c[0 * THREAD_N + 0]);

        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 1], c[0 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 2], c[0 * THREAD_N + 2]);
        
        tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 3], c[0 * THREAD_N + 3]);
        
        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 0], c[1 * THREAD_N + 0]);

        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 1], c[1 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 2], c[1 * THREAD_N + 2]);

        tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 3], c[1 * THREAD_N + 3]);

        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 0], c[2 * THREAD_N + 0]);
        
        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 1], c[2 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 2], c[2 * THREAD_N + 2]);

        tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 3], c[2 * THREAD_N + 3]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 0], c[3 * THREAD_N + 0]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 1], c[3 * THREAD_N + 1]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 2], c[3 * THREAD_N + 2]);

        tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 3], c[3 * THREAD_N + 3]);

        


        thread_prefetch = (thread_prefetch + 1) % 2;
    }
    tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 0], c[0 * THREAD_N + 0]);

    tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 1], c[0 * THREAD_N + 1]);

    tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 2], c[0 * THREAD_N + 2]);
    
    tab(a[thread_prefetch * THREAD_M + 0], b[thread_prefetch * THREAD_N + 3], c[0 * THREAD_N + 3]);
    
    tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 0], c[1 * THREAD_N + 0]);

    tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 1], c[1 * THREAD_N + 1]);

    tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 2], c[1 * THREAD_N + 2]);

    tab(a[thread_prefetch * THREAD_M + 1], b[thread_prefetch * THREAD_N + 3], c[1 * THREAD_N + 3]);

    tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 0], c[2 * THREAD_N + 0]);
    
    tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 1], c[2 * THREAD_N + 1]);

    tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 2], c[2 * THREAD_N + 2]);

    tab(a[thread_prefetch * THREAD_M + 2], b[thread_prefetch * THREAD_N + 3], c[2 * THREAD_N + 3]);

    tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 0], c[3 * THREAD_N + 0]);

    tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 1], c[3 * THREAD_N + 1]);

    tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 2], c[3 * THREAD_N + 2]);

    tab(a[thread_prefetch * THREAD_M + 3], b[thread_prefetch * THREAD_N + 3], c[3 * THREAD_N + 3]);


    
    
    #pragma unroll
    for(int j = 0; j < THREAD_N; j++){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i += 2){

            float2 tmp_1;
            tmp_1.x = c[i * THREAD_N + j].x * alpha.x - c[i * THREAD_N + j].y * alpha.y + c_load[i * THREAD_N + j].x * beta.x - c_load[i * THREAD_N + j].y * beta.y;
            tmp_1.y = c[i * THREAD_N + j].x * alpha.y + c[i * THREAD_N + j].y * alpha.x + c_load[i * THREAD_N + j].y * beta.x + c_load[i * THREAD_N + j].x * beta.y;
            
            c[i * THREAD_N + j] = tmp_1;
            
            gC[BID_Y * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i
                        + (BID_X * THREADBLOCK_N + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N + j) * M] = c[i * THREAD_N + j];


            tmp_1.x = c[(i + 1) * THREAD_N + j].x * alpha.x - c[(i + 1) * THREAD_N + j].y * alpha.y + c_load[(i + 1) * THREAD_N + j].x * beta.x - c_load[(i + 1) * THREAD_N + j].y * beta.y;
            tmp_1.y = c[(i + 1) * THREAD_N + j].x * alpha.y + c[(i + 1) * THREAD_N + j].y * alpha.x + c_load[(i + 1) * THREAD_N + j].y * beta.x + c_load[(i + 1) * THREAD_N + j].x * beta.y;
            
            c[(i + 1) * THREAD_N + j] = tmp_1;
            
            gC[BID_Y * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i + 1
                + (BID_X * THREADBLOCK_N + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N + j) * M] = c[(i + 1) * THREAD_N + j];
        }
    }
    

}