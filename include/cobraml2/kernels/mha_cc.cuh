#pragma once
#include "../algos.cuh"
#include "../macros.cuh"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

/**
 * =============================================================================
 * Multi-Head Attention using CuTe Primitives (CUDA Cores)
 * =============================================================================
 *
 * This implements the traditional 3-kernel MHA approach:
 *   1. QK Kernel:  S = Q @ K^T  (compute attention scores)
 *   2. Softmax:    P = softmax(S / sqrt(d))  (normalize scores)
 *   3. PV Kernel:  O = P @ V  (weighted sum of values)
 *
 * This is intentionally separated into 3 kernels (unlike Flash Attention which
 * fuses everything) to serve as a baseline comparison and educational example.
 *
 * Key CuTe Concepts Used:
 * -----------------------
 * - Tensor: A view over data with a Layout (shape + strides)
 * - Layout: Describes how logical coordinates map to memory offsets
 * - TiledCopy: Describes how threads cooperatively copy data (global <-> shared)
 * - TiledMMA: Describes how threads cooperatively perform matrix multiply
 * - Identity Tensor: A tensor that returns coordinates, used for predication
 *
 * Memory Hierarchy:
 * -----------------
 * Global Memory -> Shared Memory -> Registers -> Shared Memory -> Global Memory
 *      Q,K,V          Q_smem,K_smem    r_scores      (for softmax)      S,P,O
 *
 * Tensor Layout Convention:
 * -------------------------
 * - Q, K, V, O: [Batch, SeqLen, Heads, HeadDim] in global memory (BSHD format)
 * - S, P (scores): [Batch, Heads, SeqLen, SeqLen] in global memory
 * - Shared memory tiles: Row-major within each tile
 *
 * =============================================================================
 */

namespace cobraml::kernels {

using namespace cute;

namespace mha_cc_kernels {

/**
 * =============================================================================
 * Kernel 1: QK Kernel - Compute Attention Scores S = Q @ K^T
 * =============================================================================
 *
 * Grid Configuration:
 *   - gridDim.x = num_heads (H)
 *   - gridDim.y = batch_size (B)
 *   - gridDim.z = ceil(N / B_r) = number of query tile rows
 *
 * Each thread block computes one [B_r x N] strip of the attention matrix
 * by iterating over K tiles (columns of the output).
 *
 * Algorithm:
 *   1. Load Q tile [B_r x d] into shared memory (once per block)
 *   2. For each K tile [B_c x d]:
 *      a. Load K tile into shared memory
 *      b. Compute partial S = Q_tile @ K_tile^T using TiledMMA
 *      c. Scale by 1/sqrt(d) and write to global memory
 *
 * @param Q         Query tensor [B, N, H, d] in global memory
 * @param K         Key tensor [B, N, H, d] in global memory
 * @param S         Output scores [B, H, N_rows, N_cols] in global memory (padded)
 * @param N         Actual sequence length (may be < tile size)
 * @param N_rows    Padded row dimension (multiple of B_r)
 * @param N_cols    Padded column dimension (multiple of B_c)
 * @param scale     Scaling factor: 1/sqrt(head_dim)
 * @param tc        TiledCopy descriptor for global<->shared memory transfers
 * @param t_mma     TiledMMA descriptor for matrix multiplication
 */
template <typename MHAType, typename TiledCopyType, typename TiledMMAType>
__global__ void qk_kernel(const typename MHAType::TensorDType *__restrict__ Q,
                          const typename MHAType::TensorDType *__restrict__ K,
                          typename MHAType::TensorDType *__restrict__ S,
                          const int N, const int N_rows, const int N_cols,
                          const typename MHAType::TensorDType scale,
                          TiledCopyType tc, TiledMMAType t_mma) {

  using DType = typename MHAType::TensorDType;
  size_t batch_size{gridDim.y};

  // =========================================================================
  // Step 1: Create tensor views into global memory
  // =========================================================================
  // slice_head: Takes 4D tensor [B, N, H, d] and returns 2D view [N, d]
  // for this block's (batch, head) combination using blockIdx.y and blockIdx.x
  const Tensor q_head{MHAType::slice_head(Q, batch_size, N)};
  const Tensor k_head{MHAType::slice_head(K, batch_size, N)};

  // slice_scores_padded: Returns 2D view [N_rows, N_cols] of scores matrix
  // Uses padded dimensions to ensure all tile accesses are within bounds
  Tensor s_head{MHAType::slice_scores_padded(S, batch_size, N_rows, N_cols)};

  // =========================================================================
  // Step 2: Set up shared memory
  // =========================================================================
  // Dynamic shared memory is allocated at kernel launch time
  // We reinterpret it as our SharedStorage struct containing Q and K tiles
  extern __shared__ char shared_memory[];
  using SharedStorageType = typename MHAType::QKSharedStorage;
  SharedStorageType *shared_storage{
      reinterpret_cast<SharedStorageType *>(shared_memory)};

  // Create CuTe tensors backed by shared memory with specified layouts
  // QLayoutType: [B_r, d] row-major (stride = [d, 1])
  // KLayoutType: [B_c, d] row-major (stride = [d, 1])
  Tensor shared_q{make_tensor(make_smem_ptr(shared_storage->Q.begin()),
                              typename SharedStorageType::QLayoutType{})};
  Tensor shared_k{make_tensor(make_smem_ptr(shared_storage->K.begin()),
                              typename SharedStorageType::KLayoutType{})};

  // =========================================================================
  // Step 3: Define tile shapes for partitioning
  // =========================================================================
  // These are compile-time constants wrapped in CuTe's Int<> type
  constexpr typename MHAType::HeadDimType d{};      // Head dimension (e.g., 64)
  constexpr typename MHAType::BlockRowsType B_r{};  // Query tile rows (e.g., 64)
  constexpr typename MHAType::BlockColsType B_c{};  // Key tile rows (e.g., 64)

  auto q_tiler{make_shape(B_r, d)};   // Q tiles are [B_r x d]
  auto k_tiler{make_shape(B_c, d)};   // K tiles are [B_c x d]
  auto s_tiler{make_shape(B_r, B_c)}; // Output tiles are [B_r x B_c]

  // =========================================================================
  // Step 4: Create tile iterators using local_tile
  // =========================================================================
  // local_tile(tensor, tile_shape, tile_coord) creates a view that:
  //   - Divides the tensor into tiles of tile_shape
  //   - Returns a tensor where indexing gives specific tiles
  //   - Underscore (_) means "iterate over this dimension"
  //
  // q_iterator: Iterates over Q tiles along sequence dimension
  //   Shape: [B_r, d, num_q_tiles] where num_q_tiles = ceil(N/B_r)
  Tensor q_iterator{local_tile(q_head, q_tiler, make_coord(_, 0))};
  Tensor k_iterator{local_tile(k_head, k_tiler, make_coord(_, 0))};

  // s_iterator: This block writes to row tile blockIdx.z, iterates over columns
  //   Shape: [B_r, B_c, num_k_tiles]
  Tensor s_iterator{local_tile(s_head, s_tiler, make_coord(blockIdx.z, _))};

  // Number of K tiles to iterate over
  auto k_iters{size<2>(k_iterator)};

  // Get this block's Q tile (fixed for all K iterations)
  Tensor q_slice{q_iterator(_, _, blockIdx.z)};

  // =========================================================================
  // Step 5: Partition tensors for cooperative thread work
  // =========================================================================
  // TiledCopy describes how threads cooperatively move data:
  //   - Thread layout: How threads are arranged (e.g., [8, 16] = 8 rows x 16 cols)
  //   - Value layout: How many elements each thread handles per "atom"
  //
  // get_slice(threadIdx.x): Returns this thread's portion of the tiled copy
  ThrCopy thr_copy{tc.get_slice(threadIdx.x)};

  // partition_S: Partition source tensor for this thread (S = Source)
  // partition_D: Partition destination tensor for this thread (D = Destination)
  // Result shape: [values_per_copy, num_row_phases, num_col_phases]
  const Tensor tQ_global{thr_copy.partition_S(q_slice)};
  Tensor tQ_shared{thr_copy.partition_D(shared_q)};

  const Tensor tK_global_iter{thr_copy.partition_S(k_iterator)};
  Tensor tK_shared{thr_copy.partition_D(shared_k)};

  // =========================================================================
  // Step 6: Partition tensors for matrix multiply
  // =========================================================================
  // TiledMMA describes how threads cooperatively compute matrix multiply:
  //   - Each thread computes a portion of the output tile
  //   - partition_A/B: Partition input matrices for this thread
  //   - partition_C: Partition output matrix for this thread
  ThrMMA thr_mma{t_mma.get_slice(threadIdx.x)};

  Tensor q_mma{thr_mma.partition_A(shared_q)};  // A matrix (Q)
  Tensor k_mma{thr_mma.partition_B(shared_k)};  // B matrix (K, used as K^T)

  // =========================================================================
  // Step 7: Create identity tensors for predication
  // =========================================================================
  // Identity tensors return their coordinates when accessed, enabling bounds checking.
  // For a tensor sliced from 4D [batch, seq, heads, dim], the identity returns
  // a 4-tuple (b, s, h, d) where we can extract the sequence index with get<1>.
  //
  // We use N_rows (padded) to ensure the identity covers the full tile range,
  // even when tiles extend beyond the actual sequence length N.
  Tensor head_idty{MHAType::identity_slice_head_padded(batch_size, N_rows)};
  Tensor q_iterator_idty{local_tile(head_idty, q_tiler, make_coord(_, 0))};
  Tensor q_slice_idty{q_iterator_idty(_, _, blockIdx.z)};
  Tensor tQ_idty{thr_copy.partition_S(q_slice_idty)};

  Tensor k_iterator_idty{local_tile(head_idty, k_tiler, make_coord(_, 0))};
  Tensor tK_idty{thr_copy.partition_S(k_iterator_idty)};

  // =========================================================================
  // Step 8: Load Q tile (stays resident for all K iterations)
  // =========================================================================
  // predicate_copy_tensor: Copies data with bounds checking
  //   - If sequence index < N: copy from global to shared
  //   - If sequence index >= N: fill with zero (out of bounds)
  MHAType::predicate_copy_tensor(tQ_idty, tQ_global, tQ_shared, tc, DType(0), N);
  __syncthreads();  // Ensure all threads finished loading Q

  // =========================================================================
  // Step 9: Main loop - iterate over K tiles
  // =========================================================================
  for (int k_iter = 0; k_iter < k_iters; ++k_iter) {
    // Get output tile location for this iteration
    Tensor s_slice{s_iterator(_, _, k_iter)};

    // Partition output for this thread and create register fragment
    Tensor g_scores{thr_mma.partition_C(s_slice)};
    Tensor r_scores{thr_mma.make_fragment_C(g_scores)};  // Registers
    clear(r_scores);  // Initialize accumulator to zero

    // Load K tile with predication
    MHAType::predicate_copy_tensor(tK_idty(_, _, _, k_iter),
                                   tK_global_iter(_, _, _, k_iter), tK_shared,
                                   tc, DType(0), N);
    __syncthreads();

    // =========================================================================
    // Compute GEMM: r_scores += Q_shared @ K_shared^T
    // =========================================================================
    // CuTe's gemm automatically handles the transpose based on how we
    // partitioned the matrices. K is partitioned as B matrix, so the
    // MMA computes A @ B^T internally.
    gemm(t_mma, q_mma, k_mma, r_scores);

    // =========================================================================
    // Write results to global memory with predication
    // =========================================================================
    // Create 2D identity for the scores matrix to check bounds
    Tensor scores_idty{make_identity_tensor(make_shape(N_rows, N_cols))};
    Tensor scores_tile_idty{
        local_tile(scores_idty, s_tiler, make_coord(blockIdx.z, k_iter))};
    Tensor scores_mma_idty{thr_mma.partition_C(scores_tile_idty)};

    // Get dimensions of this thread's output partition
    auto write_m = size<1>(g_scores);  // Rows this thread writes
    auto write_n = size<2>(g_scores);  // Cols this thread writes

    // Write with bounds checking on both row and column
    CUTE_UNROLL
    for (int m = 0; m < write_m; ++m) {
      auto row_idx{get<0>(scores_mma_idty(0, m, 0))};
      if (row_idx >= N) continue;  // Skip invalid rows

      CUTE_UNROLL
      for (int n = 0; n < write_n; ++n) {
        auto col_idx{get<1>(scores_mma_idty(0, m, n))};
        if (col_idx < N) {
          // Scale by 1/sqrt(d) and write to global memory
          g_scores(0, m, n) = r_scores(0, m, n) * scale;
        }
      }
    }

    __syncthreads();  // Ensure shared memory can be reused
  }
}

/**
 * =============================================================================
 * Kernel 2: Softmax - Compute P = softmax(S) row-wise
 * =============================================================================
 *
 * Each thread block processes B_r rows of the attention matrix.
 * Uses warp shuffle intrinsics for efficient parallel reductions.
 *
 * Softmax Algorithm (numerically stable):
 *   1. Find max value in row (for numerical stability)
 *   2. Compute exp(x - max) for each element and sum
 *   3. Normalize by dividing by sum
 *
 * Parallel Reduction Strategy:
 *   - Each thread processes multiple elements (strided access)
 *   - Warp-level reduction using __shfl_xor_sync (no shared memory needed)
 *   - Cross-warp reduction using minimal shared memory (32 floats)
 *
 * @param S         Input scores [B, H, N_rows, N_cols]
 * @param P         Output probabilities [B, H, N_rows, N_cols]
 * @param N         Actual sequence length
 * @param N_rows    Padded row dimension
 * @param N_cols    Padded column dimension
 */
template <typename MHAType>
__global__ void softmax_kernel(typename MHAType::TensorDType *__restrict__ S,
                               typename MHAType::TensorDType *__restrict__ P,
                               const int N, const int N_rows, const int N_cols) {

  using DType = typename MHAType::TensorDType;
  size_t batch_size{gridDim.y};
  constexpr int B_r = MHAType::BlockRowsType::value;
  constexpr int threads = MHAType::threads_per_block;
  constexpr int num_warps = threads / 32;

  // Get tensor views for this block's (batch, head) combination
  Tensor s_head{MHAType::slice_scores_padded(S, batch_size, N_rows, N_cols)};
  Tensor p_head{MHAType::slice_scores_padded(P, batch_size, N_rows, N_cols)};

  int block_row_start = blockIdx.z * B_r;

  // Shared memory for cross-warp communication (only need 32 values max)
  __shared__ DType smem[32];

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  // Process each row assigned to this block
  for (int local_row = 0; local_row < B_r; ++local_row) {
    int row_idx = block_row_start + local_row;
    if (row_idx >= N) continue;  // Skip padded rows

    // Get row views
    Tensor s_row{s_head(row_idx, _)};
    Tensor p_row{p_head(row_idx, _)};

    // =========================================================================
    // Step 1: Find maximum value (for numerical stability)
    // =========================================================================
    DType local_max = -INFINITY;
    for (int j = threadIdx.x; j < N; j += threads) {
      local_max = cuda::std::max(local_max, s_row(j));
    }

    // Warp-level max reduction using shuffle
    local_max = warp_max(local_max);

    // Cross-warp reduction
    if (lane_id == 0) smem[warp_id] = local_max;
    __syncthreads();

    DType row_max;
    if (warp_id == 0) {
      DType val = (lane_id < num_warps) ? smem[lane_id] : -INFINITY;
      val = warp_max(val);
      if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();
    row_max = smem[0];

    // =========================================================================
    // Step 2: Compute exp(x - max) and sum
    // =========================================================================
    DType local_sum = 0;
    for (int j = threadIdx.x; j < N; j += threads) {
      DType exp_val = expf(s_row(j) - row_max);
      p_row(j) = exp_val;  // Store intermediate exp values
      local_sum += exp_val;
    }

    // Warp-level sum reduction
    local_sum = warp_sum(local_sum);

    // Cross-warp reduction
    if (lane_id == 0) smem[warp_id] = local_sum;
    __syncthreads();

    DType row_sum;
    if (warp_id == 0) {
      DType val = (lane_id < num_warps) ? smem[lane_id] : DType(0);
      val = warp_sum(val);
      if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();
    row_sum = smem[0];

    // =========================================================================
    // Step 3: Normalize by sum
    // =========================================================================
    DType inv_sum = DType(1) / row_sum;
    for (int j = threadIdx.x; j < N; j += threads) {
      p_row(j) *= inv_sum;
    }
  }
}

/**
 * =============================================================================
 * Kernel 3: PV Kernel - Compute Output O = P @ V
 * =============================================================================
 *
 * Similar structure to QK kernel, but computes weighted sum of values.
 *
 * Grid Configuration: Same as QK kernel
 *   - Each block computes one [B_r x d] tile of the output
 *
 * Algorithm:
 *   For each block:
 *     1. Initialize output accumulator to zero
 *     2. For each (P_tile, V_tile) pair:
 *        a. Load P tile [B_r x B_c] into shared memory
 *        b. Load V tile [B_c x d] into shared memory
 *        c. Accumulate: O += P_tile @ V_tile
 *     3. Write final output to global memory
 *
 * Key Insight - Transpose Handling:
 *   We want to compute P @ V, but CuTe's gemm computes A @ B^T.
 *   Solution: Store V in shared memory as [B_c, d], then create a
 *   "transposed view" with swapped dimensions and strides for the MMA.
 *   This gives us V^T for the MMA, so we compute P @ (V^T)^T = P @ V.
 *
 * @param P         Attention weights [B, H, N_rows, N_cols]
 * @param V         Value tensor [B, N, H, d]
 * @param O         Output tensor [B, N, H, d]
 * @param N         Actual sequence length
 * @param N_rows    Padded row dimension
 * @param N_cols    Padded column dimension
 * @param tc        TiledCopy descriptor
 * @param t_mma     TiledMMA descriptor
 */
template <typename MHAType, typename TiledCopyType, typename TiledMMAType>
__global__ void pv_kernel(const typename MHAType::TensorDType *__restrict__ P,
                          const typename MHAType::TensorDType *__restrict__ V,
                          typename MHAType::TensorDType *__restrict__ O,
                          const int N, const int N_rows, const int N_cols,
                          TiledCopyType tc, TiledMMAType t_mma) {

  using DType = typename MHAType::TensorDType;
  size_t batch_size{gridDim.y};

  // Create tensor views
  const Tensor p_head{MHAType::slice_scores_padded(P, batch_size, N_rows, N_cols)};
  const Tensor v_head{MHAType::slice_head(V, batch_size, N)};
  Tensor o_head{MHAType::slice_head(O, batch_size, N)};

  // Set up shared memory
  extern __shared__ char shared_memory[];
  using SharedStorageType = typename MHAType::PVSharedStorage;
  SharedStorageType *shared_storage{
      reinterpret_cast<SharedStorageType *>(shared_memory)};

  // P tile in shared memory: [B_r, B_c] row-major
  Tensor shared_p{make_tensor(make_smem_ptr(shared_storage->P.begin()),
                              typename SharedStorageType::PLayoutType{})};

  // V tile in shared memory: [B_c, d] row-major (for copying from global)
  Tensor shared_v{make_tensor(make_smem_ptr(shared_storage->V.begin()),
                              typename SharedStorageType::VLayoutType{})};

  // Transposed view of V for GEMM: [d, B_c] with strides [1, d]
  // This makes the MMA see V^T, so we compute P @ (V^T)^T = P @ V
  Tensor trans_shared_v{
      make_tensor(make_smem_ptr(shared_storage->V.begin()),
                  typename SharedStorageType::VTransposedLayoutType{})};

  // Tile shapes
  constexpr typename MHAType::HeadDimType d{};
  constexpr typename MHAType::BlockRowsType B_r{};
  constexpr typename MHAType::BlockColsType B_c{};

  auto p_tiler{make_shape(B_r, B_c)};  // P tiles: [B_r x B_c]
  auto v_tiler{make_shape(B_c, d)};    // V tiles: [B_c x d]
  auto o_tiler{make_shape(B_r, d)};    // O tiles: [B_r x d]

  // Create tile iterators
  // P: This block's row, iterate over columns
  Tensor p_iterator{local_tile(p_head, p_tiler, make_coord(blockIdx.z, _))};
  // V: Iterate over rows (sequence), fixed head_dim
  Tensor v_iterator{local_tile(v_head, v_tiler, make_coord(_, 0))};
  // O: This block's output tile
  Tensor o_slice{local_tile(o_head, o_tiler, make_coord(blockIdx.z, 0))};

  auto v_iters{size<2>(v_iterator)};

  // Thread partitioning
  ThrCopy thr_copy{tc.get_slice(threadIdx.x)};
  ThrMMA thr_mma{t_mma.get_slice(threadIdx.x)};

  // Output accumulator in registers
  Tensor g_out{thr_mma.partition_C(o_slice)};
  Tensor r_out{thr_mma.make_fragment_C(g_out)};
  clear(r_out);

  // MMA partitions
  Tensor p_mma{thr_mma.partition_A(shared_p)};
  Tensor v_mma{thr_mma.partition_B(trans_shared_v)};  // Note: transposed view!

  // Identity tensors for predication
  Tensor head_idty{MHAType::identity_slice_head_padded(batch_size, N_rows)};
  Tensor v_iterator_idty{local_tile(head_idty, v_tiler, make_coord(_, 0))};
  Tensor tV_idty{thr_copy.partition_S(v_iterator_idty)};

  // 2D identity for P tensor (different from 4D projection identity)
  Tensor scores_idty{make_identity_tensor(make_shape(N_rows, N_cols))};
  Tensor p_iterator_idty{
      local_tile(scores_idty, p_tiler, make_coord(blockIdx.z, _))};

  // Partition global tensors
  const Tensor tP_global_iter{thr_copy.partition_S(p_iterator)};
  Tensor tP_shared{thr_copy.partition_D(shared_p)};

  const Tensor tV_global_iter{thr_copy.partition_S(v_iterator)};
  Tensor tV_shared{thr_copy.partition_D(shared_v)};

  Tensor tP_idty_iter{thr_copy.partition_S(p_iterator_idty)};

  // =========================================================================
  // Main accumulation loop
  // =========================================================================
  for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
    // Load P tile - check both row AND column bounds (scores are 2D)
    MHAType::predicate_copy_scores(tP_idty_iter(_, _, _, v_iter),
                                   tP_global_iter(_, _, _, v_iter), tP_shared,
                                   tc, DType(0), N, N);

    // Load V tile - check sequence position bound
    MHAType::predicate_copy_tensor(tV_idty(_, _, _, v_iter),
                                   tV_global_iter(_, _, _, v_iter), tV_shared,
                                   tc, DType(0), N);
    __syncthreads();

    // Accumulate: r_out += P_shared @ V_shared
    gemm(t_mma, p_mma, v_mma, r_out);

    __syncthreads();
  }

  // =========================================================================
  // Write output with predication
  // =========================================================================
  Tensor o_iterator_idty{local_tile(head_idty, o_tiler, make_coord(blockIdx.z, 0))};
  Tensor o_mma_idty{thr_mma.partition_C(o_iterator_idty)};

  auto write_rows = size<1>(g_out);

  CUTE_UNROLL
  for (int i = 0; i < write_rows; ++i) {
    // Identity from 4D tensor: get<1> extracts sequence position
    auto seq_idx{get<1>(o_mma_idty(0, i, 0))};
    if (seq_idx < N) {
      copy(r_out(_, i, _), g_out(_, i, _));
    }
  }
}

} // namespace mha_cc_kernels

/**
 * =============================================================================
 * MHA_CC Struct - Main Interface and Configuration
 * =============================================================================
 *
 * Template Parameters:
 *   - head_count:   Number of attention heads (H)
 *   - head_dim:     Dimension of each head (d), typically 64 or 128
 *   - B_r:          Block size for query dimension (rows)
 *   - B_c:          Block size for key dimension (columns)
 *   - DType:        Data type (float, half, etc.)
 *   - thread_count: Threads per block, typically 128 or 256
 *
 * The block sizes B_r and B_c determine:
 *   - Shared memory usage: O(B_r * d + B_c * d) per block
 *   - Parallelism: More blocks = more parallelism but more overhead
 *   - Typical values: B_r = B_c = 64 or 128
 */
template <int head_count, int head_dim, int B_r, int B_c, typename DType,
          int thread_count = 128>
struct MHA_CC {

  using TensorDType = DType;
  using Self = MHA_CC<head_count, head_dim, B_r, B_c, DType, thread_count>;

  // CuTe Int<> types for compile-time dimensions
  using NumHeadsType = Int<head_count>;
  using HeadDimType = Int<head_dim>;
  using BlockRowsType = Int<B_r>;
  using BlockColsType = Int<B_c>;

  // For vectorized memory access (128-bit = 4 floats)
  using VectorizedLoadType = uint128_t;
  using ScalarLoadType = DType;

  static constexpr int threads_per_block{thread_count};
  static constexpr int elements_per_vector = sizeof(VectorizedLoadType) / sizeof(DType);

  /**
   * Shared Memory Layout for QK Kernel
   * -----------------------------------
   * Stores one Q tile and one K tile for the current iteration.
   * Both are row-major: element [i,j] at offset i*head_dim + j
   */
  struct QKSharedStorage {
    ArrayEngine<DType, B_r * head_dim> Q;  // [B_r x d]
    ArrayEngine<DType, B_c * head_dim> K;  // [B_c x d]

    // Row-major layouts
    using QLayoutType =
        Layout<Shape<BlockRowsType, HeadDimType>, Stride<HeadDimType, _1>>;
    using KLayoutType =
        Layout<Shape<BlockColsType, HeadDimType>, Stride<HeadDimType, _1>>;
  };

  /**
   * Shared Memory Layout for PV Kernel
   * -----------------------------------
   * Stores one P tile and one V tile.
   *
   * V has two views:
   *   1. VLayoutType: [B_c, d] row-major - used for copying from global
   *   2. VTransposedLayoutType: [d, B_c] - used for GEMM (gives transpose)
   *
   * The transposed view reinterprets the same memory with swapped dimensions
   * and adjusted strides, avoiding any actual data movement.
   */
  struct PVSharedStorage {
    ArrayEngine<DType, B_r * B_c> P;       // [B_r x B_c]
    ArrayEngine<DType, B_c * head_dim> V;  // [B_c x d]

    using PLayoutType =
        Layout<Shape<BlockRowsType, BlockColsType>, Stride<BlockColsType, _1>>;

    // Row-major for copying
    using VLayoutType =
        Layout<Shape<BlockColsType, HeadDimType>, Stride<HeadDimType, _1>>;

    // Transposed view for GEMM: V[j,i] accessed as B[i,j]
    // Original V[j,i] = data[j*d + i]
    // We want B[i,j] = V[j,i] = data[j*d + i], so stride = [1, d]
    using VTransposedLayoutType =
        Layout<Shape<HeadDimType, BlockColsType>, Stride<_1, HeadDimType>>;
  };

  /**
   * Layout Functions
   * ----------------
   * These create CuTe layouts that describe how tensors are stored in memory.
   * A Layout maps logical coordinates (e.g., [batch, seq, head, dim]) to
   * linear memory offsets.
   */

  // Q, K, V, O tensor layout: [Batch, SeqLen, Heads, HeadDim] row-major
  COBRA_S_DEVICE auto get_tensor_layout(size_t batch_size, size_t N) {
    return make_layout(make_shape(batch_size, N, NumHeadsType{}, HeadDimType{}),
                       LayoutRight{});  // LayoutRight = row-major
  }

  // Scores layout: [Batch, Heads, SeqLen, SeqLen]
  COBRA_S_DEVICE auto get_scores_layout(size_t batch_size, size_t N) {
    return make_layout(make_shape(batch_size, NumHeadsType{}, N, N),
                       LayoutRight{});
  }

  /**
   * Padded Scores Layout
   * --------------------
   * For unaligned sequence lengths (N % B_r != 0), we need to ensure
   * memory accesses for full tiles don't go out of bounds.
   *
   * Solution: Allocate with padded dimensions [N_rows, N_cols] but
   * only use [N, N] logically. The padding is zero-initialized.
   */
  COBRA_S_DEVICE auto get_padded_scores_layout(size_t batch_size,
                                                size_t N_rows, size_t N_cols) {
    return make_layout(
        make_shape(batch_size, NumHeadsType{}, N_rows, N_cols),
        make_stride(head_count * N_rows * N_cols, N_rows * N_cols, N_cols, _1{}));
  }

  /**
   * Tensor Slicing Functions
   * ------------------------
   * These extract 2D views from 4D tensors for the current (batch, head).
   * Using blockIdx.y for batch and blockIdx.x for head.
   */
  template <typename PtrType>
  COBRA_S_DEVICE auto slice_head(PtrType g_ptr, int batch_size, int N) {
    using BaseType = std::decay_t<std::remove_pointer_t<PtrType>>;
    static_assert(std::is_pointer_v<PtrType>, "Must be a pointer");

    const auto projection_layout{get_tensor_layout(batch_size, N)};
    const Tensor projection{
        make_tensor(make_gmem_ptr<DType>(g_ptr), projection_layout)};
    // Returns [N, d] view for this batch and head
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  template <typename PtrType>
  COBRA_S_DEVICE auto slice_scores(PtrType g_ptr, int batch_size, int N) {
    const auto scores_layout{get_scores_layout(batch_size, N)};
    const Tensor scores{
        make_tensor(make_gmem_ptr<DType>(g_ptr), scores_layout)};
    return scores(blockIdx.y, blockIdx.x, _, _);
  }

  template <typename PtrType>
  COBRA_S_DEVICE auto slice_scores_padded(PtrType g_ptr, int batch_size,
                                           int N_rows, int N_cols) {
    const auto scores_layout{get_padded_scores_layout(batch_size, N_rows, N_cols)};
    const Tensor scores{
        make_tensor(make_gmem_ptr<DType>(g_ptr), scores_layout)};
    return scores(blockIdx.y, blockIdx.x, _, _);
  }

  /**
   * Identity Tensor Functions
   * -------------------------
   * Identity tensors return their coordinates when accessed.
   * Used for bounds checking (predication) when N is not tile-aligned.
   *
   * For a 4D tensor [B, N, H, d], the identity at position (b, s, h, k)
   * returns the tuple (b, s, h, k). After slicing, we can extract
   * the sequence index with get<1>.
   */
  COBRA_S_DEVICE auto identity_slice_head(int batch_size, int N) {
    const auto projection_layout{get_tensor_layout(batch_size, N)};
    const Tensor projection{make_identity_tensor(projection_layout.shape())};
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  // Uses N_rows (padded) to ensure identity covers full tile range
  COBRA_S_DEVICE auto identity_slice_head_padded(int batch_size, int N_rows) {
    const auto projection_layout{get_tensor_layout(batch_size, N_rows)};
    const Tensor projection{make_identity_tensor(projection_layout.shape())};
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  /**
   * TiledCopy Configuration
   * -----------------------
   * Describes how threads cooperatively copy data between global and shared memory.
   *
   * Components:
   *   1. Copy_Atom: The basic copy operation (e.g., 128-bit vectorized load)
   *   2. Thread Layout: How threads are arranged [rows, cols]
   *   3. Value Layout: How many elements each thread copies per "atom"
   *
   * Example for 128 threads, head_dim=64, 4 floats per vector:
   *   - threads_per_row = 64 / 4 = 16
   *   - rows = 128 / 16 = 8
   *   - Each thread loads 4 consecutive floats (one uint128_t)
   *   - 8 row-phases needed to cover B_r=64 rows
   */
  static constexpr auto get_tiled_copy() {
    constexpr int elements_per_load{sizeof(VectorizedLoadType) / sizeof(DType)};
    constexpr int threads_per_row{head_dim / elements_per_load};

    using TPRType = Int<threads_per_row>;
    using EPLType = Int<elements_per_load>;
    constexpr int rows{thread_count / threads_per_row};
    using RowType = Int<rows>;

    static_assert(thread_count % threads_per_row == 0,
                  "thread_count is not compatible with this head dimension");
    static_assert(B_r % rows == 0,
                  "threads load too many rows, B_r must be increased");

    return make_tiled_copy(
        Copy_Atom<UniversalCopy<VectorizedLoadType>, DType>{},
        Layout<Shape<RowType, TPRType>, Stride<TPRType, _1>>{},
        Layout<Shape<_1, EPLType>>{});
  }

  // Scalar copy for cases where vectorized access isn't possible
  static constexpr auto get_scalar_tiled_copy() {
    constexpr int threads_per_row{B_c};
    constexpr int rows{thread_count / threads_per_row};
    using RowType = Int<rows>;
    using TPRType = Int<threads_per_row>;

    static_assert(thread_count % threads_per_row == 0,
                  "thread_count is not compatible with B_c for scalar copy");

    return make_tiled_copy(
        Copy_Atom<UniversalCopy<ScalarLoadType>, DType>{},
        Layout<Shape<RowType, TPRType>, Stride<TPRType, _1>>{},
        Layout<Shape<_1, _1>>{});
  }

  /**
   * TiledMMA Configuration
   * ----------------------
   * Describes how threads cooperatively compute matrix multiplication.
   *
   * For CUDA cores (no tensor cores), we use UniversalFMA which does
   * scalar multiply-add operations distributed across threads.
   *
   * Thread Layout [rows, cols]:
   *   - Determines how output elements are distributed
   *   - Each thread computes multiple output elements
   */
  static constexpr auto get_tiled_mma() {
    static_assert(thread_count % 32 == 0,
                  "thread_count must be a multiple of warp_size");

    using RowType = Int<thread_count / 32>;

    TiledMMA t_mma = make_tiled_mma(
        UniversalFMA<DType, DType, DType>{},
        Layout<Shape<RowType, _32>, Stride<_32, _1>>{});

    return t_mma;
  }

  /**
   * Predicated Copy for Projection Tensors (Q, K, V, O)
   * ---------------------------------------------------
   * Copies data with bounds checking for sequence dimension.
   *
   * The identity tensor tells us the actual sequence position.
   * Since it's from a 4D tensor [B, N, H, d], we use get<1> to
   * extract the sequence index (second component of the 4-tuple).
   *
   * @param identity_tensor  Identity for bounds checking
   * @param source_tensor    Global memory source (partitioned)
   * @param destination_tensor  Shared memory destination (partitioned)
   * @param tiled_copy       Copy descriptor
   * @param fill_value       Value to use for out-of-bounds (typically 0)
   * @param bound            Sequence length N
   */
  template <typename IdentityTensorEngineType,
            typename IdentityTensorLayoutType, typename SourceTensorEngineType,
            typename SourceTensorLayoutType,
            typename DestinationTensorEngineType,
            typename DestinationTensorLayoutType, typename TiledCopyType>
  COBRA_S_DEVICE void predicate_copy_tensor(
      const Tensor<IdentityTensorEngineType, IdentityTensorLayoutType>
          &identity_tensor,
      const Tensor<SourceTensorEngineType, SourceTensorLayoutType>
          &source_tensor,
      Tensor<DestinationTensorEngineType, DestinationTensorLayoutType>
          &destination_tensor,
      TiledCopyType tiled_copy, DType fill_value, int bound) {

    // Number of row-phases this thread handles
    constexpr int rows{size(get<1>(SourceTensorLayoutType{}))};

    CUTE_UNROLL
    for (int i{0}; i < rows; ++i) {
      // Extract sequence position from 4-tuple identity
      // Identity: (batch, seq, head, dim) -> get<1> = seq
      auto seq_idx{get<1>(identity_tensor(0, i, 0))};

      if (seq_idx < bound) {
        copy(tiled_copy, source_tensor(_, i, _), destination_tensor(_, i, _));
      } else {
        fill(destination_tensor(_, i, _), fill_value);
      }
    }
  }

  /**
   * Predicated Copy for Scores Tensor (P)
   * -------------------------------------
   * Similar to above but for 2D scores matrix [N, N].
   * Must check BOTH row and column bounds.
   *
   * The identity is a simple 2-tuple (row, col), so:
   *   - get<0> = row index
   *   - get<1> = column index
   */
  template <typename IdentityTensorEngineType,
            typename IdentityTensorLayoutType, typename SourceTensorEngineType,
            typename SourceTensorLayoutType,
            typename DestinationTensorEngineType,
            typename DestinationTensorLayoutType, typename TiledCopyType>
  COBRA_S_DEVICE void predicate_copy_scores(
      const Tensor<IdentityTensorEngineType, IdentityTensorLayoutType>
          &identity_tensor,
      const Tensor<SourceTensorEngineType, SourceTensorLayoutType>
          &source_tensor,
      Tensor<DestinationTensorEngineType, DestinationTensorLayoutType>
          &destination_tensor,
      TiledCopyType tiled_copy, DType fill_value, int row_bound, int col_bound) {

    constexpr int rows{size(get<1>(SourceTensorLayoutType{}))};

    CUTE_UNROLL
    for (int i{0}; i < rows; ++i) {
      auto row_idx{get<0>(identity_tensor(0, i, 0))};
      auto col_idx{get<1>(identity_tensor(0, i, 0))};

      if (row_idx < row_bound && col_idx < col_bound) {
        copy(tiled_copy, source_tensor(_, i, _), destination_tensor(_, i, _));
      } else {
        fill(destination_tensor(_, i, _), fill_value);
      }
    }
  }

  /**
   * Main Entry Point - operator()
   * -----------------------------
   * Orchestrates the 3-kernel MHA computation.
   *
   * Memory Management:
   *   - Allocates intermediate S and P matrices on device
   *   - Uses padded dimensions for safe tiled access
   *   - Zero-initializes padding for correct softmax behavior
   *
   * @param Q, K, V  Input tensors [B, N, H, d]
   * @param O        Output tensor [B, N, H, d]
   * @param batch_size  Number of sequences in batch
   * @param N        Sequence length
   */
  void operator()(DType *Q, DType *K, DType *V, DType *O, uint32_t batch_size,
                  uint32_t N) {

    // Compute padded dimensions for full tile coverage
    // N_rows: Round up to multiple of B_r
    // N_cols: Round up to multiple of B_c, also aligned for vectorized loads
    uint32_t N_rows = ((N + B_r - 1) / B_r) * B_r;
    uint32_t N_cols = ((N + B_c - 1) / B_c) * B_c;
    N_cols = ((N_cols + elements_per_vector - 1) / elements_per_vector) *
             elements_per_vector;

    // Allocate intermediate score matrices
    DType *S, *P;
    size_t scores_size = batch_size * head_count * N_rows * N_cols * sizeof(DType);
    cudaMalloc(&S, scores_size);
    cudaMalloc(&P, scores_size);

    // Zero-initialize - crucial for softmax correctness (padding must be 0)
    cudaMemset(S, 0, scores_size);
    cudaMemset(P, 0, scores_size);

    // Grid: [heads, batches, sequence_tiles]
    dim3 grid_dim{head_count, batch_size, ceil_div(N, B_r)};
    dim3 block_dim{thread_count};

    // Attention scale factor
    DType scale{rsqrt(static_cast<DType>(head_dim))};

    const auto tc{get_tiled_copy()};
    const auto tmma{get_tiled_mma()};

    // Kernel 1: S = Q @ K^T
    {
      auto kernel_fptr{mha_cc_kernels::qk_kernel<Self, decltype(tc), decltype(tmma)>};
      size_t smem_size{sizeof(QKSharedStorage)};

      cudaFuncSetAttribute(
          kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaFuncSetAttribute(kernel_fptr,
                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      kernel_fptr<<<grid_dim, block_dim, smem_size>>>(Q, K, S, N, N_rows, N_cols,
                                                       scale, tc, tmma);
    }

    // Kernel 2: P = softmax(S)
    {
      auto kernel_fptr{mha_cc_kernels::softmax_kernel<Self>};
      kernel_fptr<<<grid_dim, block_dim>>>(S, P, N, N_rows, N_cols);
    }

    // Kernel 3: O = P @ V
    {
      auto kernel_fptr{mha_cc_kernels::pv_kernel<Self, decltype(tc), decltype(tmma)>};
      size_t smem_size{sizeof(PVSharedStorage)};

      cudaFuncSetAttribute(
          kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      cudaFuncSetAttribute(kernel_fptr,
                           cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      kernel_fptr<<<grid_dim, block_dim, smem_size>>>(P, V, O, N, N_rows, N_cols,
                                                       tc, tmma);
    }

    cudaDeviceSynchronize();
    cudaFree(S);
    cudaFree(P);
  }
};

/**
 * Convenience wrapper function
 */
template <int H, int d>
void mha_cc_forward(float *Q, float *K, float *V, float *O, int B, int N) {
  MHA_CC<H, d, 64, 64, float, 128> mha;
  mha(Q, K, V, O, B, N);
}

} // namespace cobraml::kernels
