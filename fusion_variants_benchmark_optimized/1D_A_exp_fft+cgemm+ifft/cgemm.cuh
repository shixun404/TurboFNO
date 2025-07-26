#include <stdio.h>
#include <mma.h>
extern __shared__ float shared_mem[];
__global__ __launch_bounds__(THREAD_NUM) void cgemm(int M, int N, int K, float2 *A, float2 *B, float2 *C, float2 alpha, float2 beta){
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

    memset(c, 0, sizeof(c));
    
    int offset = 0, offset_k = 0;
    int k = 0;
    
    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_A; i++){
        tmp_A[i] = gA[BID_Y * THREADBLOCK_M + (TID + i * blockDim.x) % (THREADBLOCK_M)
        + (TID + i * blockDim.x) / THREADBLOCK_M * M];
    }

    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_B; i++){
        tmp_B[i] = gB[(TID + i * blockDim.x) % THREADBLOCK_K
        + (BID_X * THREADBLOCK_N + (TID + i * blockDim.x) / THREADBLOCK_K) * K];
    }
    
    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_A; i++){
        *(((float2*)sA) + TID + i * blockDim.x) = tmp_A[i];
    }

    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_B; i++){
        sB[(TID + i * blockDim.x) / THREADBLOCK_K + ((TID + i * blockDim.x) % THREADBLOCK_K) * THREADBLOCK_N] = ((float2*)tmp_B)[i];
    }

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < THREAD_M; i++){
        a[i] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i];
    }   
    #pragma unroll
    for(int i = 0; i < THREAD_N; i++){
        b[i] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + i];
    }
    int thread_prefetch = 0;
    int warp_prefetch = 1;
    // Main Loop along K
    #pragma unroll
    for(int k = 0; k < K - THREADBLOCK_K; k += THREADBLOCK_K){
        // if(k < K - THREADBLOCK_K){
        // Prefetech from global memory
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_A; i++){
            tmp_A[i] = gA[BID_Y * THREADBLOCK_M + (TID + i * blockDim.x) % THREADBLOCK_M
                + (TID + i * blockDim.x) / (THREADBLOCK_M) * (M) + (k + THREADBLOCK_K) * (M)];
        }
    
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_B; i++){
            tmp_B[i] = gB[(TID + i * blockDim.x) % THREADBLOCK_K + (k + THREADBLOCK_K)
                + (BID_X * THREADBLOCK_N + (TID + i * blockDim.x) / THREADBLOCK_K) * K];
        }
    // }

        // Thread-level GEMM
        #pragma unroll
        for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
            #pragma unroll
            for(int i = 0; i < THREAD_M; i++){
                a[((thread_prefetch + 1) % 2) * THREAD_M + i] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i];
            }   
            #pragma unroll
            for(int i = 0; i < THREAD_N; i++){
                b[((thread_prefetch + 1) % 2) * THREAD_N + i] = sB[(thread_k + 1) * (THREADBLOCK_N) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + i];
            }
            #pragma unroll
            for(int i = 0; i < THREAD_M; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_N; j++){
                    c[i * THREAD_N + j].x = c[i * THREAD_N + j].x + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].x;
                    c[i * THREAD_N + j].x = c[i * THREAD_N + j].x - a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].y;
                    c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].y;
                    c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].x;
                }
            }
            thread_prefetch = (thread_prefetch + 1) % 2;
        }
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++){
                c[i * THREAD_N + j].x = c[i * THREAD_N + j].x + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].x;
                c[i * THREAD_N + j].x = c[i * THREAD_N + j].x - a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].y;
                c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].y;
                c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].x;
            }
        }

        // Store prefeteched global data to shared
        // if(k < K - THREADBLOCK_K){
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_A; i++){
            *(((float2*)shared_mem_float2) + (warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K) + TID + i * blockDim.x) = tmp_A[i];
        }
    
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_B; i++){
            shared_mem_float2[THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K
                                 + (TID + i * blockDim.x) / THREADBLOCK_K + (((TID + i * blockDim.x) % THREADBLOCK_K)) * THREADBLOCK_N] = ((float2*)tmp_B)[i];
        }
                
        __syncthreads();
        
        // Prefetech from shared memory
        sA = shared_mem_float2 + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K;
        sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K;
        thread_prefetch = 0;
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            a[thread_prefetch * THREAD_M + i] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i];
        }   
        #pragma unroll
        for(int i = 0; i < THREAD_N; i++){
            b[thread_prefetch * THREAD_N + i] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N + i];
        }
        warp_prefetch = (warp_prefetch + 1) % 2;
        // }
    }


    #pragma unroll
    for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            a[((thread_prefetch + 1) % 2) * THREAD_M + i] = sA[(thread_k + 1) * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i];
        }   
        #pragma unroll
        for(int i = 0; i < THREAD_N; i++){
            b[((thread_prefetch + 1) % 2) * THREAD_N + i] = sB[(thread_k + 1) * (THREADBLOCK_N) + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + i];
        }
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++){
                c[i * THREAD_N + j].x = c[i * THREAD_N + j].x + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].x;
                c[i * THREAD_N + j].x = c[i * THREAD_N + j].x - a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].y;
                c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].y;
                c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].x;
            }
        }
        thread_prefetch = (thread_prefetch + 1) % 2;
    }
    #pragma unroll
    for(int i = 0; i < THREAD_M; i++){
        #pragma unroll
        for(int j = 0; j < THREAD_N; j++){
            c[i * THREAD_N + j].x = c[i * THREAD_N + j].x + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].x;
            c[i * THREAD_N + j].x = c[i * THREAD_N + j].x - a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].y;
            c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].x * b[thread_prefetch * THREAD_N + j].y;
            c[i * THREAD_N + j].y = c[i * THREAD_N + j].y + a[thread_prefetch * THREAD_M + i].y * b[thread_prefetch * THREAD_N + j].x;
        }
    }

    float4 tmp;
    #pragma unroll
    for(int j = 0; j < THREAD_N; j++){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i += 2){
            tmp = *((float4*)(gC + BID_Y * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i 
            + (BID_X * THREADBLOCK_N + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) %  THREAD_NUM_COL) * THREAD_N + j) * M));
            c_load[i * THREAD_N + j].x = tmp.x;
            c_load[i * THREAD_N + j].y = tmp.y;


            tmp.x = c[i * THREAD_N + j].x * alpha.x - c[i * THREAD_N + j].y * alpha.y + c_load[i * THREAD_N + j].x * beta.x - c_load[i * THREAD_N + j].y * beta.y;
            tmp.y = c[i * THREAD_N + j].x * alpha.y + c[i * THREAD_N + j].y * alpha.x + c_load[i * THREAD_N + j].y * beta.x + c_load[i * THREAD_N + j].x * beta.y;
            
            c[i * THREAD_N + j].x = tmp.x;
            c[i * THREAD_N + j].y = tmp.y;
            gC[BID_Y * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i
                        + (BID_X * THREADBLOCK_N + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N + j) * M] = c[i * THREAD_N + j];



            c_load[(i + 1) * THREAD_N + j].x = tmp.z;
            c_load[(i + 1) * THREAD_N + j].y = tmp.w;
            tmp.z = c[(i + 1) * THREAD_N + j].x * alpha.x - c[(i + 1) * THREAD_N + j].y * alpha.y + c_load[(i + 1) * THREAD_N + j].x * beta.x - c_load[(i + 1) * THREAD_N + j].y * beta.y;
            tmp.w = c[(i + 1) * THREAD_N + j].x * alpha.y + c[(i + 1) * THREAD_N + j].y * alpha.x + c_load[(i + 1) * THREAD_N + j].y * beta.x + c_load[(i + 1) * THREAD_N + j].x * beta.y;
            
            c[(i + 1) * THREAD_N + j].x = tmp.z;
            c[(i + 1) * THREAD_N + j].y = tmp.w;
            gC[BID_Y * THREADBLOCK_M + (WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i + 1
                + (BID_X * THREADBLOCK_N + (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N + j) * M] = c[(i + 1) * THREAD_N + j];
        }
    }



}