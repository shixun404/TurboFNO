#include <stdio.h>
#include <mma.h>
#define THREADBLOCK_M 64
#define THREADBLOCK_N 64
#define THREADBLOCK_K 8
#define WARP_M 32
#define WARP_N 16
#define THREAD_M 4
#define THREAD_N 4
#define WARP_NUM_ROW (THREADBLOCK_M / WARP_M)
#define THREAD_NUM_ROW (WARP_M / THREAD_M)
#define THREAD_NUM (THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N))
#define TID threadIdx.x
#define WID (threadIdx.x / 32)
#define BID_X blockIdx.x
#define BID_Y blockIdx.y
#define LOAD_PER_THREAD_A (THREADBLOCK_M * THREADBLOCK_K / THREAD_NUM)
#define LOAD_PER_THREAD_B (THREADBLOCK_N * THREADBLOCK_K / THREAD_NUM)
extern __shared__ float shared_mem[];

__global__ void fft_(int FFT_len, float2 *in, float2 *shmem){

}

__global__ void ifft(int FFT_len, float2 *in, float2 *shmem){

}

__global__ void global_to_shared(int M, int N, float2 *in, float2 *shmem){
}



__global__ void fused_fft_cgemm(int bs, int dimX, int dimY, int N, int K, float2 *A, float2 *B, float2 *C, float2 alpha, float2 beta){
    float2 *shared_mem_float2 = (float2*)shared_mem;
    float2 * gC = (float2*)C;
    float2 * gA = (float2*)A;
    float2 * gB = (float2*)B;

    float2* sA = shared_mem_float2;
    float2* sB = shared_mem_float2 + THREADBLOCK_M * THREADBLOCK_K;

    float2 c[THREAD_M][THREAD_N];
    float2 c_load[THREAD_M][THREAD_N];
    float2 a[2][THREAD_M];
    float2 b[2][THREAD_N];

    float2 tmp_A[LOAD_PER_THREAD_A];
    float2 tmp_B[LOAD_PER_THREAD_B];

    float2* regFFT[8];
    
    
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
    
    // FFT Global offest
    gPtr = gA;
    
    gPtr += TID % 16 * 1;
    
    gPtr += ((TID / 16) + threadblock_k) * bs * dimX * dimY;
    
    gPtr += (BID_X * dimY);

    fft_7(gPtr, regFFT);



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
    for(int j = 0; j < THREAD_N; j++){
        #pragma unroll
        for(int i = 0; i < THREAD_M; i++){
            gC[BID_X * THREADBLOCK_M + (WID % WARP_NUM_ROW) * WARP_M + ((TID % 32) %  THREAD_NUM_ROW) * THREAD_M + i
            + (BID_Y * THREADBLOCK_N + (WID / WARP_NUM_ROW) * WARP_N + ((TID % 32) /  THREAD_NUM_ROW) * THREAD_N + j) * M] = c[i][j];
        }
    }


}

__global__ void fft_7(float2* gPtr, float2* rPtr) {
    int threadblock_k = 0;
    
    float2* shared = (float2*) ext_shared;
    int j;
    int k;
    int global_j;
    int global_k;
    int TID;
    int offset;
    float2* gPtr;
    float2* shPtr;
    float2 rPtr_3[8];
    float2 tmp;
    float2 angle;
    float2 delta_angle;
    j = 0;
    k = -1;
    global_j = 0;
    global_k = 0;
    BID_X = blockIdx.x;
    offset = 0;
    shPtr = shared_mem;
    int bid = 0;
            
    BID_X = bid;
    
    
        rPtr[0] = *(gPtr + 0);
        rPtr_3[0].x += rPtr[0].x;
        rPtr_3[0].y += rPtr[0].y;
        
        rPtr[1] = *(gPtr + 16);
        rPtr_3[1].x += rPtr[1].x;
        rPtr_3[1].y += rPtr[1].y;
        
        rPtr[2] = *(gPtr + 32);
        rPtr_3[2].x += rPtr[2].x;
        rPtr_3[2].y += rPtr[2].y;
        
        rPtr[3] = *(gPtr + 48);
        rPtr_3[3].x += rPtr[3].x;
        rPtr_3[3].y += rPtr[3].y;
        
        rPtr[4] = *(gPtr + 64);
        rPtr_3[4].x += rPtr[4].x;
        rPtr_3[4].y += rPtr[4].y;
        
        rPtr[5] = *(gPtr + 80);
        rPtr_3[5].x += rPtr[5].x;
        rPtr_3[5].y += rPtr[5].y;
        
        rPtr[6] = *(gPtr + 96);
        rPtr_3[6].x += rPtr[6].x;
        rPtr_3[6].y += rPtr[6].y;
        
        rPtr[7] = *(gPtr + 112);
        rPtr_3[7].x += rPtr[7].x;
        rPtr_3[7].y += rPtr[7].y;
        
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[4]);
    turboFFT_ZSUB(rPtr[4], tmp, rPtr[4]);
    tmp = rPtr[4];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
        angle.x = 0.7071067811865476f;
        angle.y = -0.7071067811865475f;
        turboFFT_ZMUL(rPtr[5], tmp, angle);
        
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    rPtr[6].y = -tmp.x;
    rPtr[6].x = tmp.y;
    
    tmp = rPtr[3];
    turboFFT_ZADD(rPtr[3], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
        angle.x = -0.7071067811865475f;
        angle.y = -0.7071067811865476f;
        turboFFT_ZMUL(rPtr[7], tmp, angle);
        
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[2]);
    turboFFT_ZSUB(rPtr[2], tmp, rPtr[2]);
    tmp = rPtr[2];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    rPtr[3].y = -tmp.x;
    rPtr[3].x = tmp.y;
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    tmp = rPtr[5];
    turboFFT_ZADD(rPtr[5], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    rPtr[7].y = -tmp.x;
    rPtr[7].x = tmp.y;
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[1]);
    turboFFT_ZSUB(rPtr[1], tmp, rPtr[1]);
    tmp = rPtr[1];
    
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
    tmp = rPtr[6];
    turboFFT_ZADD(rPtr[6], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    j = 0;
    offset  = 0;
    
    offset += ((TID / 1) % 1) * 1;
    
    j = TID % 16;
    
    offset += ((TID / 1) % 2) * 8;
    
    offset += ((TID / 2) % 8) * 16;

    offset += ((TID / 128) % 4) * 128;
    
    __syncthreads();
    
    delta_angle.x = __cosf(j * -0.04908738657832146f);
    delta_angle.y = __sinf(j * -0.04908738657832146f);
     
    angle.x = 1;
    angle.y = 0;
    
    shPtr[offset + 0] = rPtr[0];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[4];
    turboFFT_ZMUL(rPtr[4], tmp, angle);
    
    shPtr[offset + 1] = rPtr[4];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[2];
    turboFFT_ZMUL(rPtr[2], tmp, angle);
    
    shPtr[offset + 2] = rPtr[2];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[6];
    turboFFT_ZMUL(rPtr[6], tmp, angle);
    
    shPtr[offset + 3] = rPtr[6];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[1];
    turboFFT_ZMUL(rPtr[1], tmp, angle);
    
    shPtr[offset + 4] = rPtr[1];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[5];
    turboFFT_ZMUL(rPtr[5], tmp, angle);
    
    shPtr[offset + 5] = rPtr[5];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[3];
    turboFFT_ZMUL(rPtr[3], tmp, angle);
    
    shPtr[offset + 6] = rPtr[3];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[7];
    turboFFT_ZMUL(rPtr[7], tmp, angle);
    
    shPtr[offset + 7] = rPtr[7];
    
    offset = 0;
    offset += TID % 16 + (TID / 16) * 128;
    
    __syncthreads();
    
    rPtr[0] = shPtr[offset + 0];
    
    rPtr[1] = shPtr[offset + 16];
    
    rPtr[2] = shPtr[offset + 32];
    
    rPtr[3] = shPtr[offset + 48];
    
    rPtr[4] = shPtr[offset + 64];
    
    rPtr[5] = shPtr[offset + 80];
    
    rPtr[6] = shPtr[offset + 96];
    
    rPtr[7] = shPtr[offset + 112];
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[4]);
    turboFFT_ZSUB(rPtr[4], tmp, rPtr[4]);
    tmp = rPtr[4];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
        angle.x = 0.7071067811865476f;
        angle.y = -0.7071067811865475f;
        turboFFT_ZMUL(rPtr[5], tmp, angle);
        
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    rPtr[6].y = -tmp.x;
    rPtr[6].x = tmp.y;
    
    tmp = rPtr[3];
    turboFFT_ZADD(rPtr[3], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
        angle.x = -0.7071067811865475f;
        angle.y = -0.7071067811865476f;
        turboFFT_ZMUL(rPtr[7], tmp, angle);
        
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[2]);
    turboFFT_ZSUB(rPtr[2], tmp, rPtr[2]);
    tmp = rPtr[2];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    rPtr[3].y = -tmp.x;
    rPtr[3].x = tmp.y;
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    tmp = rPtr[5];
    turboFFT_ZADD(rPtr[5], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    rPtr[7].y = -tmp.x;
    rPtr[7].x = tmp.y;
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[1]);
    turboFFT_ZSUB(rPtr[1], tmp, rPtr[1]);
    tmp = rPtr[1];
    
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
    tmp = rPtr[6];
    turboFFT_ZADD(rPtr[6], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    j = 0;
    offset  = 0;
    
    offset += ((TID / 1) % 4) * 1;
    
    offset += ((TID / 4) % 8) * 4;
    
    j = TID / 32;
    
    offset += ((TID / 32) % 2) * 256;
    
    __syncthreads();
    
    delta_angle.x = __cosf(j * -0.39269909262657166f);
    delta_angle.y = __sinf(j * -0.39269909262657166f);
     
    angle.x = 1;
    angle.y = 0;
    
    shPtr[offset + 0] = rPtr[0];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[4];
    turboFFT_ZMUL(rPtr[4], tmp, angle);
    
    shPtr[offset + 32] = rPtr[4];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[2];
    turboFFT_ZMUL(rPtr[2], tmp, angle);
    
    shPtr[offset + 64] = rPtr[2];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[6];
    turboFFT_ZMUL(rPtr[6], tmp, angle);
    
    shPtr[offset + 96] = rPtr[6];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[1];
    turboFFT_ZMUL(rPtr[1], tmp, angle);
    
    shPtr[offset + 128] = rPtr[1];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[5];
    turboFFT_ZMUL(rPtr[5], tmp, angle);
    
    shPtr[offset + 160] = rPtr[5];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[3];
    turboFFT_ZMUL(rPtr[3], tmp, angle);
    
    shPtr[offset + 192] = rPtr[3];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[7];
    turboFFT_ZMUL(rPtr[7], tmp, angle);
    
    shPtr[offset + 224] = rPtr[7];
    
    offset = 0;
    offset += TID;
    
    __syncthreads();
    
    rPtr[0] = shPtr[offset + 0];
    
    rPtr[1] = shPtr[offset + 64];
    
    rPtr[2] = shPtr[offset + 128];
    
    rPtr[3] = shPtr[offset + 192];
    
    rPtr[4] = shPtr[offset + 256];
    
    rPtr[5] = shPtr[offset + 320];
    
    rPtr[6] = shPtr[offset + 384];
    
    rPtr[7] = shPtr[offset + 448];
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[4]);
    turboFFT_ZSUB(rPtr[4], tmp, rPtr[4]);
    tmp = rPtr[4];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    tmp = rPtr[3];
    turboFFT_ZADD(rPtr[3], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
            
    // BID_X = bid;
    // TID = threadIdx.x;
    // gPtr = outputs;
    
    // gPtr += TID / 4 * 1;
    
    // gPtr += (BID_X % 1) * 128 * 1;
    // BID_X = BID_X / 1;
    
    // gPtr += (BID_X % 1) * 4 * 128;
    // BID_X = BID_X / 1;
    
    // gPtr += TID % 4 * 128;
    
    // gPtr += (BID_X % BS * 512);
    
            // *(gPtr + 0) = rPtr[0];            
            // *(gPtr + 16) = rPtr[1];
            // *(gPtr + 32) = rPtr[2];
            // *(gPtr + 48) = rPtr[3];
            // *(gPtr + 64) = rPtr[4];
            // *(gPtr + 80) = rPtr[5];
            // *(gPtr + 96) = rPtr[6];
            // *(gPtr + 112) = rPtr[7];
}
