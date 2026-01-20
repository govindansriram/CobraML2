#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

namespace cobraml::kernels {

using namespace cute;

// kernel configuration
template <int kBlockM = 64, int kBlockN = 64, int kBlockK = 32,
          int kHeadDim = 64>
struct MHAConfig {
  static constexpr int BlockM = kBlockM; // tile size for query sequence
  static constexpr int BlockN = kBlockN; // tile size for key/value sequence
  static constexpr int BlockK =
      kBlockK; // tile size for head dimension (for QK^T)
  static constexpr int HeadDim = kHeadDim; // head dimension d
  static constexpr int NumThreads = 128;
};

// QK^T Kernel
template <class Config>
__global__ void qk_kernel_cute(float const *Q_ptr, // [B, N, H, d] - BSHD layout
                               float const *K_ptr, // [B, N, H, d] - BSHD layout
                               float *S_ptr, // [B, H, N, N] - Output scores
                               int B, int N, int H, int d) {
  using namespace cute;

  // thread/block indices
  int bh_idx = blockIdx.x;
  int b = bh_idx / H;
  int h = bh_idx % H;
  int m_block = blockIdx.y;
  int n_block = blockIdx.z;

  // define the problem shapes
  auto shape_QK = make_shape(N, d); // (seq_len, head_dim) for Q row / K row
  auto shape_S = make_shape(N, N);  // (seq_len, seq_len) for output scores

  // strides for BSHD layout: [b,s,h,d] -> b*(N*H*d) + s*(H*d) + h*d + k
  auto stride_QK = make_stride(H * d, Int<1>{}); // (head_dim_stride, d_stride)

  // strides for S: [b,h,i,j] -> b*(H*N*N) + h*(N*N) + i*N + j
  auto stride_S = make_stride(N, Int<1>{}); // (n_stride, n_stride)

  // create global tensors for this batch/head
  // Q and K for this (b,h): offset = b*(N*H*d) + h*d
  int qk_offset = b * (N * H * d) + h * d;
  Tensor gQ = make_tensor(make_gmem_ptr(Q_ptr + qk_offset), shape_QK,
                          stride_QK); // (n, d)
  Tensor gK = make_tensor(make_gmem_ptr(K_ptr + qk_offset), shape_QK,
                          stride_QK); // (n, d)

  // s for this (b,h): offset = b*(H*N*N) + h*(N*N)
  int s_offset = b * (H * N * N) + h * (N * N);
  Tensor gS =
      make_tensor(make_gmem_ptr(S_ptr + s_offset), shape_S, stride_S); // (n, n)

  // define block tiler
  auto blk_shape = make_shape(Int<Config::BlockM>{}, Int<Config::BlockN>{},
                              Int<Config::BlockK>{});

  // get tiles for this CTA
  auto blk_coord_Q = make_coord(m_block, _); // (m_block, _)
  auto blk_coord_K = make_coord(n_block, _); // (n_block, _)
  auto blk_coord_S = make_coord(m_block, n_block);

  Tensor gQ_blk = local_tile(gQ, select<0, 2>(blk_shape),
                             blk_coord_Q); // (block_m, block_k, k_tiles)
  Tensor gK_blk = local_tile(gK, select<1, 2>(blk_shape),
                             blk_coord_K); // (block_n, block_k, k_tiles)
  Tensor gS_blk = local_tile(gS, select<0, 1>(blk_shape),
                             blk_coord_S); // (block_m, block_n)

  // shared memory for Q and K tiles
  __shared__ float smemQ[Config::BlockM * Config::BlockK];
  __shared__ float smemK[Config::BlockN * Config::BlockK];

  auto sQ_layout =
      make_layout(make_shape(Int<Config::BlockM>{}, Int<Config::BlockK>{}));
  auto sK_layout =
      make_layout(make_shape(Int<Config::BlockN>{}, Int<Config::BlockK>{}));

  Tensor sQ = make_tensor(make_smem_ptr(smemQ), sQ_layout);
  Tensor sK = make_tensor(make_smem_ptr(smemK), sK_layout);

  // thread layouts for copy and compute
  auto thr_layout_copy = make_layout(make_shape(Int<32>{}, Int<4>{}));
  auto thr_layout_mma = make_layout(make_shape(Int<16>{}, Int<8>{}));

  // partition for copy (global -> shared)
  Tensor tQgQ = local_partition(gQ_blk, thr_layout_copy,
                                threadIdx.x); // (thr_m, thr_k, k_tiles)
  Tensor tQsQ =
      local_partition(sQ, thr_layout_copy, threadIdx.x); // (thr_m, thr_k)
  Tensor tKgK = local_partition(gK_blk, thr_layout_copy,
                                threadIdx.x); // (thr_n, thr_k, k_tiles)
  Tensor tKsK =
      local_partition(sK, thr_layout_copy, threadIdx.x); // (thr_n, thr_k)

  // partition for MMA (shared memory and output)
  Tensor tCsQ = local_partition(sQ, thr_layout_mma, threadIdx.x,
                                Step<_1, X>{}); // (thr_m, block_k)
  Tensor tCsK = local_partition(sK, thr_layout_mma, threadIdx.x,
                                Step<X, _1>{}); // (thr_n, block_k)
  Tensor tCgS =
      local_partition(gS_blk, thr_layout_mma, threadIdx.x); // (thr_m, thr_n)

  // allocate accumulators in registers
  Tensor tCrS = make_tensor_like(tCgS); // (thr_m, thr_n)
  clear(tCrS);

  // scale factor
  float scale = 1.0f / sqrtf(static_cast<float>(d));

  // main loop over K dimension
  int num_k_tiles = size<2>(gQ_blk);
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    // copy Q and K tiles to shared memory
    copy(tQgQ(_, _, k_tile), tQsQ);
    copy(tKgK(_, _, k_tile), tKsK);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // compute partial dot products: S += Q @ K^T
    gemm(tCsQ, tCsK, tCrS);

    __syncthreads();
  }

  // apply scaling and write back
  for (int i = 0; i < size(tCrS); ++i) {
    tCgS(i) = tCrS(i) * scale;
  }
}

// softmax + pv kernel
template <class Config>
__global__ void
softmax_pv_kernel_cute(float const *S_ptr, // [B, H, N, N] - Attention scores
                       float const *V_ptr, // [B, N, H, d] - Values (BSHD)
                       float *O_ptr,       // [B, N, H, d] - Output (BSHD)
                       int B, int N, int H, int d) {
  using namespace cute;

  // each thread handles one (b, h, i) = one row of softmax
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * N;
  if (idx >= total)
    return;

  int b = idx / (H * N);
  int remainder = idx % (H * N);
  int h = remainder / N;
  int i = remainder % N; // query position

  // pointers for this (b, h, i)
  // S[b,h,i,:] - row i of attention matrix for this batch/head
  int s_row_offset = b * (H * N * N) + h * (N * N) + i * N;
  float const *s_row = S_ptr + s_row_offset;

  // V[b,:,h,:] - all positions, this head
  // V has stride: b*(N*H*d) + j*(H*d) + h*d + k
  int v_base = b * (N * H * d) + h * d;

  // O[b,i,h,:] - output for this position
  int o_offset = b * (N * H * d) + i * (H * d) + h * d;
  float *o_ptr = O_ptr + o_offset;

  // find max for numerical stability
  float max_val = -INFINITY;
  for (int j = 0; j < N; ++j) {
    max_val = fmaxf(max_val, s_row[j]);
  }

  // compute exp sum
  float sum_exp = 0.0f;
  for (int j = 0; j < N; ++j) {
    sum_exp += expf(s_row[j] - max_val);
  }

  // compute O = softmax(S) @ V, fused
  for (int k = 0; k < d; ++k) {
    float output = 0.0f;
    for (int j = 0; j < N; ++j) {
      float p_ij = expf(s_row[j] - max_val) / sum_exp;
      // V[b,j,h,k] = V_ptr[v_base + j*(H*d) + k]
      float v_jk = V_ptr[v_base + j * (H * d) + k];
      output += p_ij * v_jk;
    }
    o_ptr[k] = output;
  }
}

// Launcher
template <class Config = MHAConfig<>>
void mha_forward(float *Q, float *K, float *V, float *O, int B, int N, int H,
                 int d) {
  // allocate intermediate score matrix S: [B, H, N, N]
  float *S;
  cudaMalloc(&S, B * H * N * N * sizeof(float));

  // Kernel 1: QK^T
  dim3 grid_qk(B * H, (N + Config::BlockM - 1) / Config::BlockM,
               (N + Config::BlockN - 1) / Config::BlockN);
  dim3 block_qk(Config::NumThreads);

  qk_kernel_cute<Config><<<grid_qk, block_qk>>>(Q, K, S, B, N, H, d);

  // softmax + pv kernel
  int threads_softmax = 256;
  int total_rows = B * H * N;
  int blocks_softmax = (total_rows + threads_softmax - 1) / threads_softmax;

  softmax_pv_kernel_cute<Config>
      <<<blocks_softmax, threads_softmax>>>(S, V, O, B, N, H, d);

  cudaDeviceSynchronize();
  cudaFree(S);
}

} // namespace cobraml::kernels