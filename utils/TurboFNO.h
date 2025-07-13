// extern __shared__ float shared_mem[];
// #define THREADBLOCK_M 64
// #define THREADBLOCK_N 64
// #define THREADBLOCK_K 8
// #define WARP_M 32
// #define WARP_N 16
// #define THREAD_M 4
// #define THREAD_N 4
// #define WARP_NUM_ROW (THREADBLOCK_M / WARP_M)
// #define THREAD_NUM_ROW (WARP_M / THREAD_M)
// #define THREAD_NUM (THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N))
// #define TID threadIdx.x
// #define WID (threadIdx.x / 32)
// #define BID_X blockIdx.x
// #define BID_Y blockIdx.y
// #define LOAD_PER_THREAD_A (THREADBLOCK_M * THREADBLOCK_K / THREAD_NUM)
// #define LOAD_PER_THREAD_B (THREADBLOCK_N * THREADBLOCK_K / THREAD_NUM)

// int ntest=5;
// int threadblock_bs = 4;
// int threadblock_bs_1 = 8;
extern __shared__ float shared_mem[];
#define THREADBLOCK_M 64
#define THREADBLOCK_N 32
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
#define LOAD_PER_THREAD_A (THREADBLOCK_M * THREADBLOCK_K / (THREAD_NUM * 2))
#define LOAD_PER_THREAD_B (THREADBLOCK_N * THREADBLOCK_K / (THREAD_NUM * 2))

int ntest=5;
int threadblock_bs = 4;
int threadblock_bs_1 = 8;