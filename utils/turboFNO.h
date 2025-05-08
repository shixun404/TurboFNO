extern __shared__ float shared_mem[];
#define THREADBLOCK_M 64
#define THREADBLOCK_N 128
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



std::vector<int> bs_list = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768};
// std::vector<int> dimX_list = {32, 64, 128, 256};
std::vector<int> dimX_list = {1};
std::vector<int> DY_list = {128, 256};
std::vector<int> N_list = {64, 128};
std::vector<int> K_list = {8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};

int ntest=5;
int threadblock_bs = 4;