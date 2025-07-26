#include <stdio.h>
#include <mma.h>
#include "fft_radix_2_logN_7_upload_0_fused.cuh"
#include "fft_radix_2_logN_7_upload_0_fused_output.cuh"
// #include "fft_radix_2_logN_8_upload_0_fused.cuh"
// #include "fft_radix_2_logN_9_upload_0_fused.cuh"
// #include "fft_radix_2_logN_10_upload_0_fused.cuh"

__global__ __launch_bounds__(THREAD_NUM) void fused_fft_cgemm_ifft_7(int M, int N, int K, float2 *FFT_input, float2 *B, float2 *C, float2 *FFT_output, float2 alpha, float2 beta){
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (float2*)C;
    float2 * gFFT_input = (float2*)FFT_input;
    float2 * gFFT_output = (float2*)FFT_output;
    float2 * gB = (float2*)B;

    float2* sA = shared_mem_float2;
    float2* sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K;
    // float2* sFFT = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K + THREADBLOCK_N * THREADBLOCK_K;
    float2* sFFT = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_N;

    float2 c[THREAD_M * THREAD_N];
    float2 c_load[THREAD_M * THREAD_N];
    float2 a[2 * THREAD_M];
    float2 b[2 * THREAD_N];

    // float2 tmp_A[LOAD_PER_THREAD_A];
    float2 tmp_B[LOAD_PER_THREAD_B];
    
    memset(c, 0, sizeof(c));

    int offset = 0, offset_k = 0;
    int k = 0;
    
    #pragma unroll
    for(int i = 0; i < LOAD_PER_THREAD_B; i++){
        tmp_B[i] = gB[(TID + i * blockDim.x) % THREADBLOCK_K
        + (BID_X * THREADBLOCK_N + (TID + i * blockDim.x) / THREADBLOCK_K) * K];
    }
    fft_7_fused(gFFT_input + BID_Y * 128, shared_mem_float2, sFFT, M * 2);

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
    // int warp_prefetch = 1;
    // Main Loop along K
    #pragma unroll
    for(int k = 0; k < K - THREADBLOCK_K; k += THREADBLOCK_K){

        // Prefetech from global memory
        // #pragma unroll
        // for(int i = 0; i < LOAD_PER_THREAD_A; i++){
        //     tmp_A[i] = gA[BID_X * THREADBLOCK_M + (TID * LOAD_PER_THREAD_A + i) % THREADBLOCK_M
        //     + (TID * LOAD_PER_THREAD_A + i) / THREADBLOCK_M * M + (k + THREADBLOCK_K) * M];
        // }
        
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_B; i++){
            tmp_B[i] = gB[(TID + i * blockDim.x) % THREADBLOCK_K + (k + THREADBLOCK_K)
                + (BID_X * THREADBLOCK_N + (TID + i * blockDim.x) / THREADBLOCK_K) * K];
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
                    tab(a[thread_prefetch * THREAD_M + i], b[thread_prefetch * THREAD_N + j], c[i * THREAD_N + j]);
                }
            }
            thread_prefetch = (thread_prefetch + 1) % 2;
        }
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            #pragma unroll
            for(int j = 0; j < THREAD_N; j++){
                tab(a[thread_prefetch * THREAD_M + i], b[thread_prefetch * THREAD_N + j], c[i * THREAD_N + j]);
            }
        }


        __syncthreads();
        // if(threadIdx.x < THREADBLOCK_K * 128 / 8){
            fft_7_fused(gFFT_input + BID_Y * 128 + (k + THREADBLOCK_K) * M * 2, shared_mem_float2, sFFT, M * 2); 
        
        // }
        // Store prefeteched global data to shared
    
        #pragma unroll
        for(int i = 0; i < LOAD_PER_THREAD_B; i++){
            shared_mem_float2[THREADBLOCK_M * THREADBLOCK_K + (TID + i * blockDim.x) / THREADBLOCK_K + (((TID + i * blockDim.x) % THREADBLOCK_K)) * THREADBLOCK_N] = ((float2*)tmp_B)[i];
        }
        __syncthreads();
        
        // Prefetech from shared memory
        // sA = shared_mem_float2 + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K;
        // sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K + warp_prefetch * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K;
        thread_prefetch = 0;
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            a[thread_prefetch * THREAD_M + i] = sA[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + i];
        }   
        #pragma unroll
        for(int i = 0; i < THREAD_N; i++){
            b[thread_prefetch * THREAD_N + i] = sB[(WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N + i];
        }
    }
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
                tab(a[thread_prefetch * THREAD_M + i], b[thread_prefetch * THREAD_N + j], c[i * THREAD_N + j]);
            }
        }
        thread_prefetch = (thread_prefetch + 1) % 2;
    }
    #pragma unroll
    for(int i = 0; i < THREAD_M; i++){
        #pragma unroll
        for(int j = 0; j < THREAD_N; j++){
            tab(a[thread_prefetch * THREAD_M + i], b[thread_prefetch * THREAD_N + j], c[i * THREAD_N + j]);
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
        


            c_load[(i + 1) * THREAD_N + j].x = tmp.z;
            c_load[(i + 1) * THREAD_N + j].y = tmp.w;
            tmp.z = c[(i + 1) * THREAD_N + j].x * alpha.x - c[(i + 1) * THREAD_N + j].y * alpha.y + c_load[(i + 1) * THREAD_N + j].x * beta.x - c_load[(i + 1) * THREAD_N + j].y * beta.y;
            tmp.w = c[(i + 1) * THREAD_N + j].x * alpha.y + c[(i + 1) * THREAD_N + j].y * alpha.x + c_load[(i + 1) * THREAD_N + j].y * beta.x + c_load[(i + 1) * THREAD_N + j].x * beta.y;
            
            c[(i + 1) * THREAD_N + j].x = tmp.z;
            c[(i + 1) * THREAD_N + j].y = tmp.w;
        }
    }
    int threadblock_C_col = (WID % WARP_NUM_COL) * WARP_N + ((TID % 32) % THREAD_NUM_COL) * THREAD_N;
    
    __syncthreads();
    #pragma unroll
    for(int j = 0; j < THREAD_N; j++){
        // TID/4 shoule change if THREAD_M change
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            shared_mem_float2[(WID / WARP_NUM_COL) * WARP_M + ((TID % 32) /  THREAD_NUM_COL) * THREAD_M + (i + (TID / 4)) % THREAD_M
            + (threadblock_C_col + j) * THREADBLOCK_M] = c[((i + (TID / 4)) % THREAD_M) * THREAD_N + j];
        }
    }

    __syncthreads();
    #pragma unroll
    for(int tid_start = 0; tid_start < THREADBLOCK_N; tid_start += THREADBLOCK_K){
        fft_7_fused_output(shared_mem_float2 + tid_start * THREADBLOCK_M, gFFT_output + BID_Y * 128 + (BID_X * THREADBLOCK_N + tid_start) * M * 2, sFFT, M * 2);
    }



}