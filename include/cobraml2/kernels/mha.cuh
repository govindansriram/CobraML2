#pragma once
#include <cmath>
#include <cuda_runtime.h>

namespace cobraml::kernels {

// kernel 1 (QK^T):
__global__ void qk_kernel(float *Q, float *K, float *S, int B, int N, int H,
                          int d) {
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = B * H * N * N;

  if (linear_idx >= total_elements)
    return;

  int b = linear_idx / (H * N * N);
  int remainder = (linear_idx % (H * N * N));
  int h = remainder / (N * N);
  remainder = remainder % (N * N);
  int i = remainder / N; // query position
  int j = remainder % N; // key position

  // compute dot product: Q[b,i,h,:] * K[b,j,h,:]
  float score = 0.0;
  for (int k = 0; k < d; k++) {
    // BSHD: b * (N*H*d) + seq * (H*d) + h * d + k
    int q_idx = b * (N * H * d) + i * (H * d) + h * d + k;
    int k_idx = b * (N * H * d) + j * (H * d) + h * d + k;
    score += Q[q_idx] * K[k_idx];
  }

  // scale by 1/sqrt(d)
  score = score / sqrtf((float)d);

  // write - S is still [B, H, N, N] for convenience
  int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
  S[s_idx] = score;
}

// kernel 2 (Softmax):
__global__ void softmax_kernel(float *S, float *P, int B, int N, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * N;

  if (idx >= total)
    return;

  // decode which row this thread processes
  int b = idx / (H * N);
  int remainder = idx % (H * N);
  int h = remainder / N;
  int i = remainder % N;

  // find max for numerical stability
  float max_val = -INFINITY;
  for (int j = 0; j < N; j++) {
    int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    max_val = fmaxf(max_val, S[s_idx]);
  }

  // compute exp and sum
  float sum = 0.0f;
  for (int j = 0; j < N; j++) {
    int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    float exp_val = expf(S[s_idx] - max_val);
    sum += exp_val;
  }

  // write softmax
  for (int j = 0; j < N; j++) {
    int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    int p_idx = s_idx;
    float exp_val = expf(S[s_idx] - max_val);
    P[p_idx] = exp_val / sum;
  }
}

// kernel 3 (PV):
__global__ void pv_kernel(float *P, float *V, float *O, int B, int N, int H,
                          int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * N * d;

  if (idx >= total)
    return;

<<<<<<< HEAD
  // decode which output element this thread computes
  int b = idx / (H * N * d);
  int remainder = idx % (H * N * d);
  int h = remainder / (N * d);
  remainder = remainder % (N * d);
  int i = remainder / d;
  int k = remainder % d;

  // compute O[b,i,h,k] = sum_j P[b,h,i,j] * V[b,j,h,k]
  float output = 0.0f;
  for (int j = 0; j < N; j++) {
    // P is [B, H, N, N]
    int p_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    // V is BSHD: [B, N, H, d]
    int v_idx = b * (N * H * d) + j * (H * d) + h * d + k;
    output += P[p_idx] * V[v_idx];
  }

  // write - O is BSHD: [B, N, H, d]
  int o_idx = b * (N * H * d) + i * (H * d) + h * d + k;
  O[o_idx] = output;
}

// launcher
void mha_forward(float *Q, float *K, float *V, float *O, int B, int N, int H,
                 int d) {
  // Allocate intermediate buffers - S and P are [B, H, N, N]
  float *S, *P;
  cudaMalloc(&S, B * H * N * N * sizeof(float));
  cudaMalloc(&P, B * H * N * N * sizeof(float));

  // Launch kernels
  int threads = 256;
  qk_kernel<<<(B * H * N * N + threads - 1) / threads, threads>>>(Q, K, S, B, N,
                                                                  H, d);
  softmax_kernel<<<(B * H * N + threads - 1) / threads, threads>>>(S, P, B, N,
                                                                   H);
  pv_kernel<<<(B * H * N * d + threads - 1) / threads, threads>>>(P, V, O, B, N,
                                                                  H, d);

  cudaDeviceSynchronize();
  cudaFree(S);
  cudaFree(P);
}

} // namespace cobraml::kernels
=======
#pragma once
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace cobraml::kernels {

using namespace cute;

// helper functions
template <int TILE_ROW, int TILE_COL, typename DType>
CUTE_HOST_DEVICE auto make_gemm_tiled_copy() {
  return make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DType>{},
      Layout<Shape<Int<TILE_ROW>, Int<16>>>{});
}

template <int TILE_M, int TILE_N, typename DType>
CUTE_HOST_DEVICE auto make_gemm_tiled_mma() {
  return make_tiled_mma(UniversalFMA<DType, DType, DType>{},
                        Layout<Shape<Int<TILE_M>, Int<TILE_N>, _1>>{});
}

namespace mha_cute {
template <int TILE_N, int HEAD_DIM, typename DType, typename TiledCopyQ,
          typename TiledCopyK, typename TiledMMA>
__global__ void qk_kernel(const DType *__restrict__ Q, // [B, H, N, d]
                          const DType *__restrict__ K, // [B, H, N, d]
                          DType *__restrict__ S,       // [B, H, N, N]
                          int B, int H, int N, TiledCopyQ tiled_copy_q,
                          TiledCopyK tiled_copy_k, TiledMMA tiled_mma) {
  // Create global memory tensor
  auto Q_tensor = make_tensor(
      make_gmem_ptr(Q),
      make_layout(make_shape(B, H, N, Int<HEAD_DIM>{}), LayoutRight{}));
  auto K_tensor = make_tensor(
      make_gmem_ptr(K),
      make_layout(make_shape(B, H, N, Int<HEAD_DIM>{}), LayoutRight{}));
  auto S_tensor = make_tensor(
      make_gmem_ptr(S), make_layout(make_shape(B, H, N, N), LayoutRight{}));

  // Decode batch and head indices from blockIdx.z
  int bh = blockIdx.z;
  int b_idx = bh / H;
  int h_idx = bh % H;

  // Slice tensors to current batch and head
  auto Q_bh = Q_tensor(b_idx, h_idx, _, _); // [N, HEAD_DIM]
  auto K_bh = K_tensor(b_idx, h_idx, _, _); // [N, HEAD_DIM]
  auto S_bh = S_tensor(b_idx, h_idx, _, _); // [N, N]

  // cta tile coordinates
  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  // extract this CTA's tiles
  auto gQ = local_tile(Q_bh, make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}),
                       make_coord(tile_row, 0));

  auto gK = local_tile(K_bh, make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}),
                       make_coord(tile_col, 0));

  auto gS = local_tile(S_bh, make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                       make_coord(tile_row, tile_col));

  // shared memory layouts
  auto sQ_layout =
      make_layout(make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}), LayoutRight{});
  auto sK_layout =
      make_layout(make_shape(Int<TILE_N>{}, Int<HEAD_DIM>{}), LayoutRight{});

  __shared__ DType smem_q[cosize_v<decltype(sQ_layout)>];
  __shared__ DType smem_k[cosize_v<decltype(sK_layout)>];

  // create shared memory tensor views
  Tensor sQ = make_tensor(make_smem_ptr(smem_q), sQ_layout);
  Tensor sK = make_tensor(make_smem_ptr(smem_k), sK_layout);

  // thread partitioning for copy operations
  auto thr_copy_q = tiled_copy_q.get_slice(threadIdx.x);
  auto thr_copy_k = tiled_copy_k.get_slice(threadIdx.x);

  // partition source (global) and destination (shared) tensors
  Tensor tQgQ = thr_copy_q.partition_S(gQ);
  Tensor tQsQ = thr_copy_q.partition_D(sQ);
  Tensor tKgK = thr_copy_k.partition_S(gK);
  Tensor tKsK = thr_copy_k.partition_D(sK);

  // thread partitioning for mma operations
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  // partition shared memory for mma consumption
  Tensor tCsQ = thr_mma.partition_A(sQ);
  Tensor tCsK = thr_mma.partition_B(sK);
  Tensor tCgS = thr_mma.partition_C(gS);

  // allocate and clear accumulator fragment
  Tensor tCrS = thr_mma.make_fragment_C(tCgS);
  clear(tCrS);

  // load Q and K tiles to shared memory
  copy(tiled_copy_q, tQgQ, tQsQ);
  copy(tiled_copy_k, tKgK, tKsK);
  __syncthreads();

  // compute gemm: S = Q @ K^T
  gemm(tiled_mma, tCsQ, tCsK, tCrS);

  // apply scaling factor 1/sqrt(d) and write to global memory
  DType scale = DType(1.0) / sqrt(DType(HEAD_DIM));
  CUTE_UNROLL
  for (int i = 0; i < size(tCrS); ++i) {
    tCrS(i) *= scale;
  }

  copy(tCrS, tCgS);
}

template <int BLOCK_SIZE, typename DType>
__global__ void softmax_kernel(const DType *__restrict__ S, // [B, H, N, N]
                               DType *__restrict__ P,       // [B, H, N, N]
                               int B, int H, int N) {
  // each block handles one row
  int row_idx = blockIdx.x;
  int total_rows = B * H * N;
  if (row_idx >= total_rows)
    return;

  // decode (b, h, i) from flattened row index
  int b = row_idx / (H * N);
  int rem = row_idx % (H * N);
  int h = rem / N;
  int i = rem % N;

  // create tensor views
  auto S_tensor = make_tensor(
      make_gmem_ptr(S), make_layout(make_shape(B, H, N, N), LayoutRight{}));
  auto P_tensor = make_tensor(
      make_gmem_ptr(P), make_layout(make_shape(B, H, N, N), LayoutRight{}));

  // get this row
  auto S_row = S_tensor(b, h, i, _);
  auto P_row = P_tensor(b, h, i, _);

  // shared memory for parallel reduction
  __shared__ DType smax[BLOCK_SIZE];
  __shared__ DType ssum[BLOCK_SIZE];

  int tid = threadIdx.x;

  // find row maximum
  DType thread_max = -INFINITY;
  for (int j = tid; j < N; j += BLOCK_SIZE) {
    thread_max = fmaxf(thread_max, S_row(j));
  }
  smax[tid] = thread_max;
  __syncthreads();

  // parallel reduction to find global max
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
    }
    __syncthreads();
  }
  DType row_max = smax[0];

  // compute exp(x - max) and sum
  DType thread_sum = DType(0);
  for (int j = tid; j < N; j += BLOCK_SIZE) {
    thread_sum += expf(S_row(j) - row_max);
  }
  ssum[tid] = thread_sum;
  __syncthreads();

  // parallel reduction to find sum
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      ssum[tid] += ssum[tid + stride];
    }
    __syncthreads();
  }
  DType row_sum = ssum[0];

  // write normalized softmax values
  for (int j = tid; j < N; j += BLOCK_SIZE) {
    P_row(j) = expf(S_row(j) - row_max) / row_sum;
  }
}

template <int TILE_N, int TILE_D, typename DType, typename TiledCopyP,
          typename TiledCopyV, typename TiledMMA>
__global__ void pv_kernel(const DType *__restrict__ P, // [B, H, N, N]
                          const DType *__restrict__ V, // [B, H, N, d]
                          DType *__restrict__ O,       // [B, H, N, d]
                          int B, int H, int N, int d, TiledCopyP tiled_copy_p,
                          TiledCopyV tiled_copy_v, TiledMMA tiled_mma) {
  // create global memory tensor views
  auto P_tensor = make_tensor(
      make_gmem_ptr(P), make_layout(make_shape(B, H, N, N), LayoutRight{}));
  auto V_tensor = make_tensor(
      make_gmem_ptr(V), make_layout(make_shape(B, H, N, d), LayoutRight{}));
  auto O_tensor = make_tensor(
      make_gmem_ptr(O), make_layout(make_shape(B, H, N, d), LayoutRight{}));

  // decode batch and head indices
  int bh = blockIdx.z;
  int b_idx = bh / H;
  int h_idx = bh % H;

  // slice to current batch and head
  auto P_bh = P_tensor(b_idx, h_idx, _, _); // [N, N]
  auto V_bh = V_tensor(b_idx, h_idx, _, _); // [N, d]
  auto O_bh = O_tensor(b_idx, h_idx, _, _); // [N, d]

  // cta tile coordinates
  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  // shared memory layouts
  auto sP_layout =
      make_layout(make_shape(Int<TILE_N>{}, Int<TILE_N>{}), LayoutRight{});
  auto sV_layout =
      make_layout(make_shape(Int<TILE_N>{}, Int<TILE_D>{}), LayoutRight{});

  __shared__ DType smem_p[cosize_v<decltype(sP_layout)>];
  __shared__ DType smem_v[cosize_v<decltype(sV_layout)>];

  Tensor sP = make_tensor(make_smem_ptr(smem_p), sP_layout);
  Tensor sV = make_tensor(make_smem_ptr(smem_v), sV_layout);

  // thread partitioning for copy
  ThrCopy thr_copy_p = tiled_copy_p.get_slice(threadIdx.x);
  ThrCopy thr_copy_v = tiled_copy_v.get_slice(threadIdx.x);

  // thread partitioning for mma
  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

  // number of tiles along the reduction dimension
  int num_j_tiles = (N + TILE_N - 1) / TILE_N;

  // get output tile and allocate accumulator
  auto gO = local_tile(O_bh, make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
                       make_coord(tile_row, tile_col));

  Tensor tCgO = thr_mma.partition_C(gO);
  Tensor tCrO = thr_mma.make_fragment_C(tCgO);
  clear(tCrO);

  // accumulation loop over N dimension
  for (int j_tile = 0; j_tile < num_j_tiles; ++j_tile) {
    // get P tile: [TILE_N, TILE_N]
    auto gP = local_tile(P_bh, make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                         make_coord(tile_row, j_tile));

    // get V tile: [TILE_N, TILE_D]
    auto gV = local_tile(V_bh, make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
                         make_coord(j_tile, tile_col));

    // Partition for copy
    Tensor tPgP = thr_copy_p.partition_S(gP);
    Tensor tVgV = thr_copy_v.partition_S(gV);

    Tensor tPsP = thr_copy_p.partition_D(sP);
    Tensor tVsV = thr_copy_v.partition_D(sV);

    // Load P and V tiles to shared memory
    copy(tiled_copy_p, tPgP, tPsP);
    copy(tiled_copy_v, tVgV, tVsV);

    __syncthreads();

    // Partition shared memory for MMA
    Tensor tCsP = thr_mma.partition_A(sP);
    Tensor tCsV = thr_mma.partition_B(sV);

    // Compute P @ V and accumulate
    gemm(tiled_mma, tCsP, tCsV, tCrO);

    __syncthreads();
  }

  copy(tCrO, tCgO);
}
} // namespace mha_cute

template <int TILE_N = 16, int TILE_D = 16, int SOFTMAX_BLOCK = 256,
          typename DType = float>
void mha_forward(DType *Q, DType *K, DType *V, DType *O, int B, int H, int N,
                 int d) {
  // allocate intermediate buffers
  DType *S, *P;
  cudaMalloc(&S, B * H * N * N * sizeof(DType));
  cudaMalloc(&P, B * H * N * N * sizeof(DType));

  {
    auto tiled_copy_q = make_gemm_tiled_copy<TILE_N, TILE_D, DType>();
    auto tiled_copy_k = make_gemm_tiled_copy<TILE_N, TILE_D, DType>();
    auto tiled_mma = make_gemm_tiled_mma<TILE_N, TILE_N, DType>();

    constexpr int num_threads = TILE_N * TILE_N;

    dim3 grid((N + TILE_N - 1) / TILE_N, (N + TILE_N - 1) / TILE_N, B * H);
    dim3 block(num_threads);

    mha_cute::qk_kernel<TILE_N, TILE_D, DType><<<grid, block>>>(
        Q, K, S, B, H, N, tiled_copy_q, tiled_copy_k, tiled_mma);
  }

  {
    int total_rows = B * H * N;
    mha_cute::softmax_kernel<SOFTMAX_BLOCK, DType>
        <<<total_rows, SOFTMAX_BLOCK>>>(S, P, B, H, N);
  }

  {
    auto tiled_copy_p = make_gemm_tiled_copy<TILE_N, TILE_N, DType>();
    auto tiled_copy_v = make_gemm_tiled_copy<TILE_N, TILE_D, DType>();
    auto tiled_mma = make_gemm_tiled_mma<TILE_N, TILE_D, DType>();

    constexpr int num_threads = TILE_N * TILE_D;

    dim3 grid((d + TILE_D - 1) / TILE_D, (N + TILE_N - 1) / TILE_N, B * H);
    dim3 block(num_threads);

    mha_cute::pv_kernel<TILE_N, TILE_D, DType><<<grid, block>>>(
        P, V, O, B, H, N, d, tiled_copy_p, tiled_copy_v, tiled_mma);
  }

  cudaDeviceSynchronize();
  cudaFree(S);
  cudaFree(P);
}

} // namespace cobraml::kernels
>>>>>>> 21fcf3a (move mha forward outside and some minor fixes)
