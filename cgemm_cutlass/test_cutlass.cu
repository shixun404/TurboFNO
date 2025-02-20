#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cuh"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "helper.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_complex.h"



#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_reduce.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
using DataT = cutlass::complex<float>;
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;

  int m, n, k;
  double l2_norm_3xtf32_vs_fp64;
  double l2_norm_1xtf32_vs_fp64;
  double l2_norm_fp32_vs_fp64;

  // ctor
  Result(
    int m, int n, int k,
    double runtime_ms, double gflops,
    double l2_norm_3xtf32_vs_fp64,
    double l2_norm_1xtf32_vs_fp64,
    double l2_norm_fp32_vs_fp64) :
    m(m), n(n), k(k),
    runtime_ms(runtime_ms), gflops(gflops),
    l2_norm_3xtf32_vs_fp64(l2_norm_3xtf32_vs_fp64),
    l2_norm_1xtf32_vs_fp64(l2_norm_1xtf32_vs_fp64),
    l2_norm_fp32_vs_fp64(l2_norm_fp32_vs_fp64)   {}

  Result() {}
};

bool CutlassCgemmNN(int num_tests, int M, int N, int K, DataT alpha, DataT const *A, int lda, DataT const *B, int ldb, DataT beta, DataT *C,int ldc) {
  // using ElementAccumulator = DataT;
  // using ElementComputeEpilogue = ElementAccumulator;
  // using ElementInputA = DataT;
  // using ElementInputB = DataT;
  // using ElementOutput = DataT;

  // using LayoutInputA = cutlass::layout::ColumnMajor;
  // using LayoutInputB = cutlass::layout::ColumnMajor;
  // using LayoutOutput = cutlass::layout::ColumnMajor;

  // using MMAOp = cutlass::arch::OpClassSimt;

  // using SmArch = cutlass::arch::Sm80;

  // using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 8>;
  // using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 16, 8>;
  // using ShapeMMAOp = cutlass::gemm::GemmShape<1, 1, 1>;

  // using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
  //     ElementOutput,
  //     1,
  //     ElementAccumulator,
  //     ElementComputeEpilogue>; 

  // constexpr int NumStages = 2;

  // using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
  //                                          LayoutInputA,
  //                                          ElementInputB,
  //                                          LayoutInputB,
  //                                          ElementOutput,
  //                                          LayoutOutput,
  //                                          ElementAccumulator,
  //                                          MMAOp,
  //                                          SmArch,
  //                                          ShapeMMAThreadBlock,
  //                                          ShapeMMAWarp,
  //                                          ShapeMMAOp,
  //                                          EpilogueOp,
  //                                          SwizzleThreadBlock,
  //                                          NumStages>;

  using precision = cutlass::complex<float>;
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
    using WarpShape = cutlass::gemm::GemmShape<16, 32, 8>;

    static int const kEpilogueElementsPerAccess = 1;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision, kEpilogueElementsPerAccess, precision, precision>;

    using Gemm = cutlass::gemm::device::Gemm<
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision, cutlass::layout::ColumnMajor,
        precision,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2 // Stages
    >;

  Gemm gemm_op;

  int split_k_slices = 1;
  cutlass::gemm::GemmCoord problem_size({M, N, K});
  ////////////////////////////////////////////////////////////////////////////////
  /// 3. Run  3xTF32 kernel within a profiling loop
  ////////////////////////////////////////////////////////////////////////////////
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
    {A, lda},
    {B, ldb},
    {C, ldc},
    {C, ldc},
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);


  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Result structure
  Result result;

  //
  // Construct events
  //

  cudaEvent_t events[2];

  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return false;
    }
  }

  // Record an event at the start of a series of GEMMs
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return false;
  }

  //
  // Run profiling loop
  //
  gemm_op();
  for (int iter = 0; iter < num_tests; ++iter) {
    // Launch initialized CUTLASS kernel
    status = gemm_op();
    cudaDeviceSynchronize();
    // CUTLASS_CHECK(status);
  }

  //
  // Stop profiling loop
  //

  // Record an event when the GEMMs are complete
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return false;
  }

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return false;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return false;
  }

  // Compute average runtime and GFLOPs.
  result.m = problem_size.m();
  result.n = problem_size.n();
  result.k = problem_size.k();
  result.runtime_ms = double(runtime_ms) / double(num_tests);
  result.gflops =  (2.0 *  double(problem_size.product()) / double(1.0e9)) / (result.runtime_ms / 1000.0);
  printf("cutlass: %d, %d, %d, %f, %f\n", result.m, result.n, result.k, result.runtime_ms, result.gflops);
  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // // CUTLASS_CHECK(status);
  // if (status != cutlass::Status::kSuccess)
  //   return 0;
  return 1;
}

void test_cutlass(int m, int n, int k, int num_tests, DataT alpha, DataT beta, DataT * dA, DataT *dB, DataT *dC){
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    bool ret = CutlassCgemmNN(num_tests, m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    // if (ret == 0) return;
    // cudaDeviceSynchronize();
    // cudaEventRecord(beg);
    // for(int i = 0; i < num_tests; ++i){
    //     CutlassCgemmNN(m, n, k, alpha, dA, m, dB, k, beta, dC, m);
    //     cudaDeviceSynchronize();
    // }
    // cudaEventRecord(end);
    // cudaEventSynchronize(beg);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&elapsed, beg, end);
    // double gflops = (double(2 * num_tests * double(m) * double(n) * double(k)) / (1e9)) / (elapsed / 1e3);
    // printf("%d, %d, %d, %f, %f\n", m, n, k, elapsed, gflops);
}

int main(int argc, char** argv){
    DataT *A, *dA, *B, *dB, *C, *C_ref, *dC, *dC_ref;
    int M, N, K;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    // freopen("input.txt", "r", stdin);
    // scanf("%d%d%d", &M, &N, &K);
    long long int A_size = ((M + 127) / 128) * 128 * ((K + 127) / 128) * 128;
    long long int B_size = ((N + 127) / 128) * 128 * ((K + 127) / 128) * 128;
    long long int C_size = ((M + 127) / 128) * 128 * ((N + 127) / 128) * 128;
    A = (DataT*)malloc(sizeof(DataT) * A_size);
    B = (DataT*)malloc(sizeof(DataT) * B_size);
    C = (DataT*)malloc(sizeof(DataT) * C_size);
    C_ref = (DataT*)malloc(sizeof(DataT) * C_size);

    cudaMalloc((void**)&dA, sizeof(DataT) * A_size);
    cudaMalloc((void**)&dB, sizeof(DataT) * B_size);
    cudaMalloc((void**)&dC, sizeof(DataT) * C_size);
    cudaMalloc((void**)&dC_ref, sizeof(DataT) * C_size);

    for(long long int i = 0; i < A_size; ++i) {
      A[i].real() = float(rand() % 5) + (rand() % 5) * 0.01;
      A[i].imag() = float(rand() % 5) + (rand() % 5) * 0.01;
    }
    for(long long int i = 0; i < B_size; ++i){
      B[i].real() = float(rand() % 5) + (rand() % 5) * 0.01;
      B[i].imag() = float(rand() % 5) + (rand() % 5) * 0.01;
    }
    for(long long int i = 0; i < C_size; ++i){
      C[i].real() = float(rand() % 5) + (rand() % 5) * 0.01;
      C[i].imag() = float(rand() % 5) + (rand() % 5) * 0.01;
    }

    cudaMemcpy(dA, A, sizeof(DataT) * A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(DataT) * B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC_ref, C, sizeof(DataT) * C_size, cudaMemcpyHostToDevice);

    DataT alpha = {0.1,0.1} , beta = {0.1,0.1}; 

    int num_tests = argc > 4 ? atoi(argv[4]) : 1;

    // test_cutlass(M, N, K, num_tests, alpha, beta, dA, dB, dC);
    
    cublasHandle_t handle;   
    cublasCreate(&handle);
    cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);     
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    float elapsed;
    cudaDeviceSynchronize();
    cudaEventRecord(beg);
    cudaDeviceSynchronize();
    for(int i = 0; i < num_tests; ++i){
      cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (cuFloatComplex*)&alpha, (cuFloatComplex*)dA, M, (cuFloatComplex*)dB, K, (cuFloatComplex*)&beta, (cuFloatComplex*)dC_ref, M);
      cudaDeviceSynchronize();
    }  
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, beg, end);
    double gflops = (double(2 * num_tests * double(M) * double(N) * double(K)) / (1e9)) / (elapsed / 1e3);
    printf("cublas %d, %d, %d, %f, %f\n", M, N, K, elapsed, gflops);

    test_cutlass(M, N, K, num_tests, alpha, beta, dA, dB, dC);
    
    cudaMemcpy(C, dC, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref, dC_ref, sizeof(DataT) * C_size, cudaMemcpyDeviceToHost);

    verify_vector((float*)C_ref, (float*)C, M * N * 2);

    return 0;
}