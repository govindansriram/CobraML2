#pragma once
#include "../algos.cuh"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

// Fused Multi Head Attention, that runs purely on cuda cores
// Based on the Flash Attention 1 algorithm.

namespace cobraml::kernels {

using namespace cute;

namespace naive {

// num_heads, batch_size, ceil(N / B_r)

/**
 * @brief performs naive gpu mha
 *
 * @tparam num_heads an equal sized chunk of projection
 * this allows the model to localize attention to that specific chunk
 * hence multi head attention.
 * @tparam d the length of each head
 * @param Q the query projection [batch, num_heads, N, d]
 * @param K the key projection [batch, num_heads, N, d]
 * @param V the value projection [batch, num_heads, N, d]
 * @param O the output [batch, num_heads, N, d]
 * @param N sequence length: tokens per sequence
 * @return void
 */
template <typename MHAType, typename TiledCopyQType, typename TiledCopyKType,
          typename TiledCopyVType, typename TiledMMAQK, typename TiledMMAPV>
__global__ void
mha_kernel(const typename MHAType::TensorDType *__restrict__ Q,
           const typename MHAType::TensorDType *__restrict__ K,
           const typename MHAType::TensorDType *__restrict__ V,
           typename MHAType::TensorDType *__restrict__ O, const int N,
           const typename MHAType::TensorDType scale, TiledCopyQType tc_q,
           TiledCopyKType tc_k, TiledCopyVType tc_v, TiledMMAQK t_mma_qk,
           TiledMMAPV t_mma_pv) {

  using DType = typename MHAType::TensorDType;

  auto q_ptr{make_gmem_ptr<DType>(Q)};
  auto k_ptr{make_gmem_ptr<DType>(K)};
  auto v_ptr{make_gmem_ptr<DType>(V)};
  auto o_ptr{make_gmem_ptr<DType>(O)};

  size_t batch_size{gridDim.y};

  const auto q_layout{MHAType::get_tensor_layout(batch_size, N)};
  const auto k_layout{MHAType::get_tensor_layout(batch_size, N)};
  const auto v_layout{MHAType::get_tensor_layout(batch_size, N)};
  const auto o_layout{MHAType::get_tensor_layout(batch_size, N)};

  const auto q_tensor{make_tensor(q_ptr, q_layout)};
  const auto k_tensor{make_tensor(k_ptr, k_layout)};
  const auto v_tensor{make_tensor(v_ptr, v_layout)};
  const auto o_tensor{make_tensor(o_ptr, o_layout)};

  const auto q_head{q_tensor(blockIdx.y, blockIdx.x, _, _)};
  const auto k_head{k_tensor(blockIdx.y, blockIdx.x, _, _)};
  const auto v_head{v_tensor(blockIdx.y, blockIdx.x, _, _)};
  const auto o_head{o_tensor(blockIdx.y, blockIdx.x, _, _)};

  extern __shared__ char shared_memory[];
  using SharedStorageType = typename MHAType::SharedStorage;
  SharedStorageType *shared_storage{
      reinterpret_cast<SharedStorageType *>(shared_memory)};

  auto shared_q_ptr{make_smem_ptr(shared_storage->Q.begin())};
  auto shared_k_ptr{make_smem_ptr(shared_storage->K.begin())};
  auto shared_v_ptr{make_smem_ptr(shared_storage->V.begin())};
  auto shared_p_ptr{make_smem_ptr(shared_storage->P.begin())};
  auto shared_o_ptr{make_smem_ptr(shared_storage->O.begin())};

  Tensor shared_q{
      make_tensor(shared_q_ptr, typename SharedStorageType::QOLayoutType{})};

  Tensor shared_k{
      make_tensor(shared_k_ptr, typename SharedStorageType::KVLayoutType{})};

  Tensor shared_v{
      make_tensor(shared_k_ptr, typename SharedStorageType::KVLayoutType{})};

  Tensor trans_shared_v{make_tensor(
      shared_k_ptr, typename SharedStorageType::VTransposedLayoutType{})};

  Tensor shared_p{
      make_tensor(shared_p_ptr, typename SharedStorageType::PLayoutType{})};

  // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

  constexpr typename MHAType::HeadDimType d{};
  constexpr typename MHAType::QueryRowsType B_r{};
  constexpr typename MHAType::KVColsType B_c{};

  auto coord{make_coord(_, 0)};

  auto q_tiler{make_shape(B_r, d)};
  auto kv_tiler{make_shape(B_c, d)};

  Tensor q_iterator{
      local_tile(q_head, q_tiler, coord)}; // (B_r, d, floor(N / B_r))
  Tensor k_iterator{
      local_tile(k_head, kv_tiler, coord)}; // (B_c, d, floor(N / B_c))
  Tensor v_iterator{
      local_tile(v_head, kv_tiler, coord)}; // (B_c, d, floor(N / B_c))
  Tensor o_iterator{
      local_tile(o_head, q_tiler, coord)}; // (B_c, d, floor(N / B_c))

  auto iters{size<2>(k_iterator)};

  Tensor q_slice{q_iterator(_, _, blockIdx.z)};
  Tensor out_slice{o_iterator(_, _, blockIdx.z)};

  // move q to shared memory

  // t prefix means unique to this thread
  ThrCopy thr_copy_q{tc_q.get_slice(threadIdx.x)};
  ThrCopy thr_copy_k{tc_k.get_slice(threadIdx.x)};
  ThrCopy thr_copy_v{tc_v.get_slice(threadIdx.x)};

  Tensor tQ_global_part{thr_copy_q.partition_S(q_slice)};
  Tensor tQ_shared_part{thr_copy_q.partition_D(shared_q)};
  Tensor tK_shared_part{thr_copy_k.partition_D(shared_k)};
  Tensor tV_shared_part{thr_copy_v.partition_D(shared_v)};

  ThrMMA thr_mma_qk{t_mma_qk.get_slice(threadIdx.x)};

  Tensor q_mma{thr_mma_qk.partition_A(shared_q)};
  Tensor k_mma{thr_mma_qk.partition_B(shared_k)};
  Tensor p_mma{thr_mma_qk.partition_C(shared_p)};

  ThrMMA thr_mma_pv{t_mma_pv.get_slice(threadIdx.x)};

  Tensor p_mma2{thr_mma_pv.partition_A(shared_p)};
  Tensor v_mma{thr_mma_pv.partition_B(trans_shared_v)};
  Tensor g_out_mma{thr_mma_pv.partition_C(out_slice)};
  Tensor r_out_mma{thr_mma_pv.make_fragment_C(g_out_mma)};

  auto mma_m{select<1>(q_mma.shape())};

  Tensor r_scores_mma{thr_mma_qk.make_fragment_C(p_mma)};
  clear(r_scores_mma); // Zero the accumulator

  copy(tc_q, tQ_global_part, tQ_shared_part);
  __syncthreads(); // redundant

  // start with the lowest possible value
  Tensor m{make_tensor<DType>(mma_m)};
  Tensor l{make_tensor<DType>(mma_m)};
  fill(m, cuda::std::numeric_limits<DType>::lowest());
  fill(l, static_cast<DType>(0));

  for (size_t iter{0}; iter < iters; ++iter) {
    Tensor k_slice{k_iterator(_, _, iter)};
    Tensor v_slice{v_iterator(_, _, iter)};

    Tensor tK_global_part{thr_copy_k.partition_S(k_slice)};
    Tensor tV_global_part{thr_copy_v.partition_S(v_slice)};

    copy(tc_k, tK_global_part, tK_shared_part);

    __syncthreads();

    gemm(t_mma_qk, q_mma, k_mma, r_scores_mma);
    MHAType::update_statistics(m, l, r_scores_mma, p_mma, r_out_mma, scale);
    copy(tc_v, tV_global_part, tV_shared_part);

    __syncthreads();

    gemm(t_mma_pv, p_mma2, v_mma, r_out_mma);
  }

  constexpr Layout mma_shape{get<0>(r_out_mma.layout())};
  constexpr size_t m_rows{size(get<1>(r_out_mma.layout()))};

  static_assert(rank(mma_shape) == 1,
                "only rank 1 mma shape is currently supported");

  CUTE_UNROLL
  for (size_t m_row{0}; m_row < m_rows; ++m_row) {
    auto out_slice{r_out_mma(_, m_row, _)};

    CUTE_UNROLL
    for (size_t idx{0}; idx < size(out_slice); ++idx) {
      out_slice(idx) = out_slice(idx) / l(m_row);
    }
  }

  // scale

  copy(r_out_mma, g_out_mma);
}
} // namespace naive

/**
 * @brief
 *
 * @tparam head_count
 * @tparam head_dim the length of each head
 * @tparam B_r how many sequences of q to process at once
 * @tparam B_c how many sequences of K and V to process at once
 * @tparam DType
 * @tparam thread_count
 */
template <int head_count, int head_dim, int B_r, int B_c, typename DType,
          int thread_count = 128>
struct FMHA {

  using TensorDType = DType;
  using Self = FMHA<head_count, head_dim, B_r, B_c, DType, thread_count>;

  using NumHeadsType = Int<head_count>;
  using HeadDimType = Int<head_dim>;
  using QueryRowsType = Int<B_r>;
  using KVColsType = Int<B_c>;

  using VectorizedLoadType = uint128_t;

  struct SharedStorage {
    ArrayEngine<DType, B_r * head_dim> Q;
    ArrayEngine<DType, B_c * head_dim> K;
    ArrayEngine<DType, B_c * head_dim> V;
    ArrayEngine<DType, B_r * B_c> P;
    ArrayEngine<DType, B_r * head_dim> O;

    using QOLayoutType =
        Layout<Shape<QueryRowsType, HeadDimType>, Stride<HeadDimType, _1>>;
    using KVLayoutType =
        Layout<Shape<KVColsType, HeadDimType>, Stride<HeadDimType, _1>>;
    using PLayoutType =
        Layout<Shape<KVColsType, QueryRowsType>, Stride<QueryRowsType, _1>>;
    using VTransposedLayoutType = Layout<Shape<HeadDimType, KVColsType>>;
  };

  static constexpr int threads_per_block{thread_count};

  CUTE_HOST_DEVICE static auto get_tensor_layout(size_t batch_size, size_t N) {
    return make_layout(make_shape(batch_size, NumHeadsType{}, N, HeadDimType{}),
                       LayoutRight{});
  }

  CUTE_HOST_DEVICE static constexpr int get_total_threads() {
    constexpr int elements_per_load{sizeof(VectorizedLoadType) / sizeof(DType)};
    constexpr int threads_per_row{head_dim / elements_per_load};

    static_assert(head_dim % elements_per_load == 0,
                  "head dimension in bytes must be a multiple of 128");

    constexpr int total_threads{B_r * threads_per_row};

    static_assert(total_threads % 32 == 0,
                  "block size requires an odd number of threads");

    return total_threads;
  }

  static constexpr auto get_tiled_copy() {

    // ensures no repeat across the head dimension

    constexpr int elements_per_load{sizeof(VectorizedLoadType) / sizeof(DType)};
    constexpr int threads_per_row{head_dim / elements_per_load};

    static_assert(head_dim % threads_per_row == 0,
                  "the head dimension is not a multiple of 128 bytes");

    using TPRType = Int<threads_per_row>;
    using EPLType = Int<elements_per_load>;
    constexpr int rows{thread_count / threads_per_row};
    using RowType = Int<rows>;

    static_assert(thread_count % threads_per_row == 0,
                  "thread_count is not compatible with this head dimension");

    static_assert(B_r % rows == 0,
                  "threads load too many rows B_r must be increased");

    return make_tiled_copy(
        Copy_Atom<UniversalCopy<VectorizedLoadType>, DType>{},
        Layout<Shape<RowType, TPRType>, Stride<TPRType, _1>>{},
        Layout<Shape<_1, EPLType>>{});
  }

  static constexpr auto get_tiled_mma() {

    static_assert(thread_count % 32 == 0,
                  "thread_count must be a multiple of warp_size");

    using RowType = Int<thread_count / 32>;

    // one warp computes one row

    TiledMMA t_mma = make_tiled_mma(
        UniversalFMA<DType, DType, DType>{},
        Layout<Shape<RowType, _32>, Stride<_32, _1>>{}); // 16x16x1 UniversalFMA

    return t_mma;
  }

  static_assert(B_c % B_r == 0, "B_c must be a multiple of B_r");

  template <typename MaxTensorEngineType, typename RScoresTensorEngineType,
            typename ProbTensorEngineType, typename OutTensorEngineType,
            typename MaxTensorLayoutType, typename ScoresTensorLayoutType,
            typename ProbTensorLayoutType, typename OutTensorLayoutType>
  __device__ inline static void update_statistics(
      Tensor<MaxTensorEngineType, MaxTensorLayoutType> &max_tensor,
      Tensor<MaxTensorEngineType, MaxTensorLayoutType> &sum_tensor,
      Tensor<RScoresTensorEngineType, ScoresTensorLayoutType> &r_scores,
      Tensor<ProbTensorEngineType, ProbTensorLayoutType> &prob_tensor,
      Tensor<OutTensorEngineType, OutTensorLayoutType> &out_tensor,
      const DType scale) {

    static_assert(rank_v<ScoresTensorLayoutType> == 3,
                  "Per Register Attention scores must be 3 dimensional (mma, "
                  "mma_m, mma_n)");

    static_assert(rank_v<MaxTensorLayoutType> == 1,
                  "Per register, row maxes, muse be 1 dimensional");

    constexpr size_t mma_m{size(get<1>(ScoresTensorLayoutType{}))};
    constexpr Layout mma_shape{get<0>(ScoresTensorLayoutType{})};

    CUTE_UNROLL
    for (size_t m{0}; m < mma_m; ++m) {

      auto r_score_slice{r_scores(_, m, _)};
      auto p_slice{(prob_tensor(_, m, _))};
      auto o_slice{(out_tensor(_, m, _))};

      if constexpr (rank(mma_shape) > 1)
        static_assert(rank(mma_shape) > 1, "not yet implemented");

      auto &current_max{max_tensor(m)};
      auto old_max{current_max};
      auto &current_sum{sum_tensor(m)};

      constexpr size_t slice_size{size(r_score_slice)};

      CUTE_UNROLL
      for (size_t idx{0}; idx < slice_size; ++idx) {
        // uses hardware unit, removes warp divergence from branch checks
        // each thread may hold multiple values from each row, we find the
        // local maximum first
        r_score_slice(idx) = r_score_slice(idx) * scale; // scale by 1 / sqrt(d)
        current_max = cuda::std::max(r_score_slice(idx), current_max);
      }

      current_max = warp_max(current_max);

      // Compute scaling factor for old values
      auto scale_old = expf(old_max - current_max);

      // scale the sum
      current_sum = current_sum * scale_old;

      DType local_sum{0};

      CUTE_UNROLL
      for (size_t idx{0}; idx < slice_size; ++idx) {
        auto p_score{r_score_slice(idx)};
        p_score = expf(p_score - current_max);
        local_sum += p_score;
        // write to probs tensor
        p_slice(idx) = p_score;
        // reset registers
        r_score_slice(idx) = 0;
      }

      current_sum += warp_sum(local_sum);

      CUTE_UNROLL
      for (size_t i{0}; i < size(o_slice); i++) {
        o_slice(i) *= scale_old;
      }
    }
  }

  void operator()(DType *Q, DType *K, DType *V, DType *O, uint32_t batch_size,
                  uint32_t N) {
    dim3 grid_dim{head_count, batch_size, ceil_div(N, B_r)};

    dim3 block_dim{thread_count};

    DType scale{rsqrt(static_cast<DType>(head_dim))};

    const auto tc_q{get_tiled_copy()};
    const auto tc_k{get_tiled_copy()};
    const auto tc_v{get_tiled_copy()};
    const auto qk_mma{get_tiled_mma()};
    const auto pv_mma{get_tiled_mma()};

    auto kernel_fptr{
        naive::mha_kernel<Self, decltype(tc_q), decltype(tc_k), decltype(tc_v),
                          decltype(qk_mma), decltype(pv_mma)>};

    size_t smem_size{sizeof(SharedStorage)};

    // Set L1 to be SMEM only
    cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaFuncSetAttribute(kernel_fptr,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<grid_dim, block_dim, smem_size>>>(Q, K, V, O, N, scale, tc_q,
                                                    tc_k, tc_v, qk_mma, pv_mma);

    // print(tc_q);
    // print_latex(tc_q);
  }
};

} // namespace cobraml::kernels