#include <cobraml2/kernels/matmul_fp4.cuh>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

using namespace cobraml;
using namespace cute;

template <int N, int K, int MMA_M, int MMA_N, int pipeline_stages>
void test_gemm(int M) {

  using AType = cutlass::bfloat16_t;
  using BType = cutlass::bfloat16_t;
  using CType = float;

  using GemmType = kernels::SM100::GEMM<AType, BType, CType>;
  GemmType::Config<N, K, MMA_M, MMA_N, pipeline_stages> config;

  // Allocate device memory
  AType * d_a;
  BType * d_b;
  CType * d_c;

  size_t a_bytes{M * K * sizeof(AType)};
  size_t b_bytes{N * K * sizeof(BType)};
  size_t c_bytes{M * N * sizeof(CType)};

  cudaMalloc(&d_a, a_bytes);
  cudaMalloc(&d_b, b_bytes);
  cudaMalloc(&d_c, c_bytes);

  cudaMemset(d_a, 0, a_bytes);
  cudaMemset(d_b, 0, b_bytes);
  cudaMemset(d_c, 0, c_bytes);

  GemmType gemm;
  gemm(d_a, d_b, d_c, M, config);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

TEST(GEMM_SM100, BF16_M256_N512_K512) { test_gemm<512, 512, 128, 256, 2>(256); }