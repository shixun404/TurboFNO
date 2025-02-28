#include <stdio.h>
#include <mma.h>
// #include "fft_radix_2_logN_7_upload_0_fused.cuh"
#include "fft_radix_2_logN_7_upload_0_fused_output.cuh"
// #include "fft_radix_2_logN_8_upload_0_fused.cuh"
// #include "fft_radix_2_logN_9_upload_0_fused.cuh"
// #include "fft_radix_2_logN_10_upload_0_fused.cuh"

__global__ void fused_cgemm_ifft_7(int M, int N, int K, float2 *A, float2 *B, float2 *C, float2 *FFT_output, float2 alpha, float2 beta){
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (float2*)C;
    float2 * gFFT_output = (float2*)FFT_output;
    float2 * gB = (float2*)B;
    float2 * gA = (float2*)A;

    float2* sA = shared_mem_float2;
    float2* sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K;
    float2* sFFT = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K + THREADBLOCK_N * THREADBLOCK_K;

    float2 c[THREAD_M][THREAD_N];
    float2 c_load[THREAD_M][THREAD_N];
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
        tmp_B[i] = gB[(TID * LOAD_PER_THREAD_B + i) / THREADBLOCK_N
        + (BID_Y * THREADBLOCK_N + (TID * LOAD_PER_THREAD_B + i) % THREADBLOCK_N) * K];
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
        a[0][i] = sA[(WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i];
    }   
    #pragma unroll
    for(int i = 0; i < THREAD_N; i++){
        b[0][i] = sB[(WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + i];
    }
    int thread_prefetch = 0;
    int warp_prefetch = 1;
    // Main Loop along K
    #pragma unroll
    for(int k = 0; k < K - THREADBLOCK_K; k += THREADBLOCK_K){

        // Prefetech from global memory
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_A; i++){
            tmp_A[i] = gA[BID_X * THREADBLOCK_M + (TID * LOAD_PER_THREAD_A + i) % THREADBLOCK_M
            + (TID * LOAD_PER_THREAD_A + i) / THREADBLOCK_M * M + (k + THREADBLOCK_K) * M];
        }
    
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_B; i++){
            tmp_B[i] = gB[(TID * LOAD_PER_THREAD_B + i) / THREADBLOCK_N + (k + THREADBLOCK_K)
            + (BID_Y * THREADBLOCK_N + (TID * LOAD_PER_THREAD_B + i) % THREADBLOCK_N) * K];
        }

        // Thread-level GEMM
        #pragma unroll
        for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
            #pragma unroll
            for(int i = 0; i < THREAD_M; i++){
                a[(thread_prefetch + 1) % 2][i] = sA[(thread_k + 1) * THREADBLOCK_M + (WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i];
            }   
            #pragma unroll
            for(int i = 0; i < THREAD_N; i++){
                b[(thread_prefetch + 1) % 2][i] = sB[(thread_k + 1) * THREADBLOCK_N + (WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + i];
            }
            #pragma unroll
            for(int i = 0; i < THREAD_M; i++){
                #pragma unroll
                for(int j = 0; j < THREAD_N; j++){
                    c[i][j].x += a[thread_prefetch][i].x * b[thread_prefetch][j].x - a[thread_prefetch][i].y * b[thread_prefetch][j].y;
                    c[i][j].y += a[thread_prefetch][i].x * b[thread_prefetch][j].y + a[thread_prefetch][i].y * b[thread_prefetch][j].x;
                }
            }
            thread_prefetch = (thread_prefetch + 1) % 2;
        }
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++){
                c[i][j].x += a[thread_prefetch][i].x * b[thread_prefetch][j].x - a[thread_prefetch][i].y * b[thread_prefetch][j].y;
                c[i][j].y += a[thread_prefetch][i].x * b[thread_prefetch][j].y + a[thread_prefetch][i].y * b[thread_prefetch][j].x;
            }
        }

        // Store prefeteched global data to shared
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_A; i++){
            shared_mem_float2[warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K + TID * LOAD_PER_THREAD_A + i] = tmp_A[i];
        }
    
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_B; i++){
            shared_mem_float2[THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K
                                 + TID * LOAD_PER_THREAD_B + i] = tmp_B[i];
        }
        
        __syncthreads();
        
        // Prefetech from shared memory
        sA = shared_mem_float2 + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K;
        sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K;
        thread_prefetch = 0;
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            a[thread_prefetch][i] = sA[(WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i];
        }   
        #pragma unroll
        for(int i = 0; i < THREAD_N; i++){
            b[thread_prefetch][i] = sB[(WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + i];
        }
        warp_prefetch = (warp_prefetch + 1) % 2;
    }
    // Thread-level GEMM
    #pragma unroll
    for(int thread_k = 0; thread_k < THREADBLOCK_K - 1; ++thread_k){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            a[(thread_prefetch + 1) % 2][i] = sA[(thread_k + 1) * THREADBLOCK_M + (WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i];
        }   
        #pragma unroll
        for(int i = 0; i < THREAD_N; i++){
            b[(thread_prefetch + 1) % 2][i] = sB[(thread_k + 1) * THREADBLOCK_N + (WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + i];
        }
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++){
                c[i][j].x += a[thread_prefetch][i].x * b[thread_prefetch][j].x - a[thread_prefetch][i].y * b[thread_prefetch][j].y;
                c[i][j].y += a[thread_prefetch][i].x * b[thread_prefetch][j].y + a[thread_prefetch][i].y * b[thread_prefetch][j].x;
            }
        }
        thread_prefetch = (thread_prefetch + 1) % 2;
    }
    #pragma unroll
    for(int i = 0; i < THREAD_M; i++){
        #pragma unroll
        for(int j = 0; j < THREAD_N; j++){
            c[i][j].x += a[thread_prefetch][i].x * b[thread_prefetch][j].x - a[thread_prefetch][i].y * b[thread_prefetch][j].y;
            c[i][j].y += a[thread_prefetch][i].x * b[thread_prefetch][j].y + a[thread_prefetch][i].y * b[thread_prefetch][j].x;
        }
    }


    #pragma unroll
    for(int j = 0; j < THREAD_N; j++){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            c_load[i][j] = gC[BID_X * THREADBLOCK_M + (WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i
            + (BID_Y * THREADBLOCK_N + (WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + j) * M];
        }
    }


    #pragma unroll
    for(int j = 0; j < THREAD_N; j++){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            float2 tmp;
            tmp.x = c[i][j].x * alpha.x - c[i][j].y * alpha.y + c_load[i][j].x * beta.x - c_load[i][j].y * beta.y;
            tmp.y = c[i][j].x * alpha.y + c[i][j].y * alpha.x + c_load[i][j].y * beta.x + c_load[i][j].x * beta.y;
            c[i][j] = tmp;
        }
    }


    #pragma unroll
    for(int tid_start = 0; tid_start < THREADBLOCK_N; tid_start += THREADBLOCK_K){
        int threadblock_C_col = (WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N;
        __syncthreads();
        if(tid_start <= threadblock_C_col && threadblock_C_col < tid_start + THREADBLOCK_K) {
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++){
                // TID/4 shoule change if THREAD_M change
                #pragma unroll
                for(int i = 0; i < THREAD_M; i++){
                    sFFT[(WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + (i + (TID / 4)) % THREAD_M
                    + (threadblock_C_col + j - tid_start) * 128] = c[(i + (TID / 4)) % THREAD_M][j];
                }
            }
        }
        __syncthreads();
        fft_7_fused_output(sFFT, gFFT_output + BID_X * 128 + (BID_Y * THREADBLOCK_N + tid_start) * M * 2, sFFT, M * 2);
    }


}