#pragma once
#include "../algos.cuh"
#include "../macros.cuh"
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
 * @param Q the query projection [batch, N, num_heads, d]
 * @param K the key projection [batch, N, num_heads, d]
 * @param V the value projection [batch, N, num_heads, d]
 * @param O the output [batch, N, num_heads, d]
 * @param N sequence length: tokens per sequence
 * @return void
 */
template <typename MHAType, typename TiledCopyTypeQK, typename TiledCopyTypeV,
          typename TiledMMAType>
__global__ void
mha_kernel(const typename MHAType::TensorDType *__restrict__ Q,
           const typename MHAType::TensorDType *__restrict__ K,
           const typename MHAType::TensorDType *__restrict__ V,
           typename MHAType::TensorDType *__restrict__ O, const int N,
           const typename MHAType::TensorDType scale, TiledCopyTypeQK tc_qk,
           TiledCopyTypeV tc_v, TiledMMAType t_mma) {

  using DType = typename MHAType::TensorDType;
  size_t batch_size{gridDim.y};

  const Tensor q_head{MHAType::slice_head(Q, batch_size, N)};
  const Tensor k_head{MHAType::slice_head(K, batch_size, N)};
  const Tensor v_head{MHAType::slice_head(V, batch_size, N)};
  Tensor o_head{MHAType::slice_head(O, batch_size, N)};

  extern __shared__ char shared_memory[];
  using SharedStorageType = typename MHAType::SharedStorage;
  SharedStorageType *shared_storage{
      reinterpret_cast<SharedStorageType *>(shared_memory)};

  Tensor shared_q{make_tensor(make_smem_ptr(shared_storage->Q.begin()),
                              typename SharedStorageType::QLayoutType{})};

  Tensor shared_k{make_tensor(make_smem_ptr(shared_storage->KV.begin()),
                              typename SharedStorageType::KLayoutType{})};

  Tensor shared_v{make_tensor(make_smem_ptr(shared_storage->KV.begin()),
                              typename SharedStorageType::VLayoutType{})};

  Tensor trans_shared_v{
      make_tensor(make_smem_ptr(shared_storage->KV.begin()),
                  typename SharedStorageType::VTransposedLayoutType{})};

  Tensor shared_p{make_tensor(make_smem_ptr(shared_storage->P.begin()),
                              typename SharedStorageType::PLayoutType{})};

  // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

  constexpr typename MHAType::HeadDimType d{};
  constexpr typename MHAType::QueryRowsType B_r{};
  constexpr typename MHAType::KVColsType B_c{};

  auto coord{make_coord(_, 0)};

  auto q_tiler{make_shape(B_r, d)};
  auto kv_tiler{make_shape(B_c, d)};
  auto scores_tiler{make_shape(B_r, B_c)};

  Tensor q_iterator{
      local_tile(q_head, q_tiler, coord)}; // (B_r, d, ceil(N / B_r))
  Tensor k_iterator{
      local_tile(k_head, kv_tiler, coord)}; // (B_c, d, ceil(N / B_c))
  Tensor v_iterator{
      local_tile(v_head, kv_tiler, coord)}; // (B_c, d, ceil(N / B_c))
  Tensor o_iterator{
      local_tile(o_head, q_tiler, coord)}; // (B_c, d, ceil(N / B_r))

  auto iters{size<2>(k_iterator)};

  Tensor q_slice{q_iterator(_, _, blockIdx.z)};
  Tensor out_slice{o_iterator(_, _, blockIdx.z)};

  // t prefix means unique to this thread
  ThrCopy thr_copy_qk{tc_qk.get_slice(threadIdx.x)};
  ThrCopy thr_copy_v{tc_v.get_slice(threadIdx.x)};

  const Tensor tQ_global_part{thr_copy_qk.partition_S(q_slice)};
  Tensor tQ_shared_part{thr_copy_qk.partition_D(shared_q)};

  const Tensor tK_global_part_iter{thr_copy_qk.partition_S(k_iterator)};
  Tensor tK_shared_part{thr_copy_qk.partition_D(shared_k)};

  const Tensor tV_global_part_iter{thr_copy_v.partition_S(v_iterator)};
  Tensor tV_shared_part{thr_copy_v.partition_D(shared_v)};

  ThrMMA thr_mma_qk{t_mma.get_slice(threadIdx.x)};

  Tensor q_mma{thr_mma_qk.partition_A(shared_q)};
  Tensor k_mma{thr_mma_qk.partition_B(shared_k)};
  Tensor p_mma{thr_mma_qk.partition_C(shared_p)};

  Tensor p_mma2{thr_mma_qk.partition_A(shared_p)};
  Tensor v_mma{thr_mma_qk.partition_B(trans_shared_v)};
  Tensor g_out_mma{thr_mma_qk.partition_C(out_slice)};
  Tensor r_out_mma{thr_mma_qk.make_fragment_C(g_out_mma)};

  Tensor head_idty{MHAType::identity_slice_head(batch_size, N)};

  // make q identity tensor
  Tensor q_iterator_idty{
      local_tile(head_idty, q_tiler, coord)}; // (B_r, d, ceil(N / B_r))
  Tensor q_head_slice_idty{q_iterator_idty(_, _, blockIdx.z)};
  Tensor tQ_idty_part{thr_copy_qk.partition_S(q_head_slice_idty)};

  // make k identity tensor
  Tensor kv_iterator_idty{local_tile(head_idty, kv_tiler, coord)};
  Tensor k_idty_part{thr_copy_qk.partition_S(kv_iterator_idty)};

  // make v identity tensor
  Tensor v_idty_part{thr_copy_v.partition_S(kv_iterator_idty)};

  // scores identity tensor
  Tensor scores_idty{make_identity_tensor(make_shape(N, N))};
  Tensor scores_tile_idty{
      local_tile(scores_idty, scores_tiler,
                 make_coord(blockIdx.z, _))}; // (B_r, B_c, ceil(N / B_c))
  Tensor scores_slice_idty{thr_mma_qk.partition_C(scores_tile_idty)};

  // out identity tensor
  Tensor o_iterator_idty{
      local_tile(head_idty, q_tiler, coord)}; // (B_r, d, ceil(N / B_r))
  Tensor o_head_slice_idty{o_iterator_idty(_, _, blockIdx.z)};
  Tensor o_mma_idty{thr_mma_qk.partition_C(o_head_slice_idty)};

  // predicated copy
  MHAType::predicate_copy_tensor(tQ_idty_part, tQ_global_part, tQ_shared_part,
                                 tc_qk, DType(0), N);

  auto mma_m{select<1>(q_mma.shape())};

  Tensor r_scores_mma{thr_mma_qk.make_fragment_C(p_mma)};
  clear(r_scores_mma); // Zero the accumulator

  // start with the lowest possible value
  Tensor m{make_tensor<DType>(mma_m)};
  Tensor l{make_tensor<DType>(mma_m)};
  fill(m, -INFINITY);
  clear(l); // zero the sums

  // Do the block that needs predication first

  MHAType::matmul<true>(tK_global_part_iter(_, _, _, iters - 1), tK_shared_part,
                        tc_qk, q_mma, k_mma, r_scores_mma,
                        k_idty_part(_, _, _, iters - 1), t_mma, N);

  MHAType::update_statistics<true>(m, l, r_scores_mma, p_mma, r_out_mma,
                                   scores_slice_idty(_, _, _, iters - 1), scale,
                                   N);

  __syncthreads(); // ensure all K reads done before V overwrites KV buffer
  MHAType::matmul<true>(tV_global_part_iter(_, _, _, iters - 1), tV_shared_part,
                        tc_v, p_mma2, v_mma, r_out_mma,
                        v_idty_part(_, _, _, iters - 1), t_mma, N);

  // do the rest of blocks that don't need predication
  for (int iter{static_cast<int>(iters) - 2}; iter > -1; --iter) {
    __syncthreads();
    MHAType::matmul(tK_global_part_iter(_, _, _, iter), tK_shared_part, tc_qk,
                    q_mma, k_mma, r_scores_mma, k_idty_part(_, _, _, iter),
                    t_mma, N);

    MHAType::update_statistics(m, l, r_scores_mma, p_mma, r_out_mma,
                               scores_slice_idty(_, _, _, iter), scale, N);

    __syncthreads(); // ensure all K reads done before V overwrites KV buffer
    MHAType::matmul(tV_global_part_iter(_, _, _, iter), tV_shared_part, tc_v,
                    p_mma2, v_mma, r_out_mma, v_idty_part(_, _, _, iter), t_mma,
                    N);
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

  constexpr int write_rows{size(get<1>(g_out_mma.shape()))};

  CUTE_UNROLL
  for (size_t i{0}; i < write_rows; ++i) {
    auto seq_idx{get<1>(o_mma_idty(0, i, 0))};

    if (seq_idx < N)
      copy(r_out_mma(_, i, _), g_out_mma(_, i, _));
  }
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
          int thread_count = 128, bool causal_mask = false>
struct FMHA {

  using TensorDType = DType;
  using Self =
      FMHA<head_count, head_dim, B_r, B_c, DType, thread_count, causal_mask>;

  using NumHeadsType = Int<head_count>;
  using HeadDimType = Int<head_dim>;
  using QueryRowsType = Int<B_r>;
  using KVColsType = Int<B_c>;

  using VectorizedLoadType = uint128_t;
  using ScalarLoadType = uint32_t;

  struct SharedStorage {
    // Swizzle atom dimensions
    static constexpr int kSwizzleAtomRows = 8;
    static constexpr int kSwizzleAtomCols = 32;

    // Static asserts for tiling compatibility
    static_assert(
        B_r % kSwizzleAtomRows == 0,
        "B_r must be divisible by 8 (swizzle atom row size) for Q layout");
    static_assert(
        B_c % kSwizzleAtomCols == 0,
        "B_c must be divisible by 32 (swizzle atom col size) for V layout");
    static_assert(head_dim % kSwizzleAtomCols == 0,
                  "head_dim must be divisible by 32 (swizzle atom col size) "
                  "for Q/K layouts");

    ArrayEngine<DType, B_r * head_dim> Q;
    ArrayEngine<DType, B_c * head_dim> KV;
    // ArrayEngine<DType, B_c * head_dim> V;
    ArrayEngine<DType, B_r * B_c> P;

    using swizzle_atom = decltype(composition(
        Swizzle<3, 2, 3>{},
        Layout<Shape<_8, Shape<_4, _8>>, Stride<_32, Stride<_1, _4>>>{}));

    using swizzle_atom_T = decltype(composition(
        Swizzle<3, 2, 3>{},
        Layout<Shape<Shape<_4, _8>, _8>, Stride<Stride<_1, _4>, _32>>{}));

    using QLayoutType = decltype(tile_to_shape(
        swizzle_atom{}, make_shape(QueryRowsType{}, HeadDimType{})));
    using KLayoutType = decltype(tile_to_shape(
        swizzle_atom{}, make_shape(KVColsType{}, HeadDimType{})));
    using VTransposedLayoutType = decltype(tile_to_shape(
        swizzle_atom{}, make_shape(HeadDimType{}, KVColsType{})));
    using VLayoutType = decltype(tile_to_shape(
        swizzle_atom_T{}, make_shape(KVColsType{}, HeadDimType{}),
        LayoutRight{}));
    using PLayoutType =
        Layout<Shape<KVColsType, QueryRowsType>, Stride<QueryRowsType, _1>>;
  };

  static constexpr int threads_per_block{thread_count};

  COBRA_S_DEVICE auto get_tensor_layout(size_t batch_size, size_t N) {
    return make_layout(make_shape(batch_size, N, NumHeadsType{}, HeadDimType{}),
                       LayoutRight{});
  }

  template <typename PtrType>
  COBRA_S_DEVICE auto slice_head(PtrType g_ptr, int batch_size, int N) {

    using BaseType = std::decay_t<std::remove_pointer_t<PtrType>>;
    static_assert(std::is_pointer_v<PtrType>, "Must be a pointer");
    static_assert(
        std::is_same_v<std::remove_cv_t<std::remove_pointer_t<PtrType>>, DType>,
        "Must point to DType");

    const auto projection_layout{get_tensor_layout(batch_size, N)};
    const Tensor projection{
        make_tensor(make_gmem_ptr<DType>(g_ptr), projection_layout)};
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  COBRA_S_DEVICE auto identity_slice_head(int batch_size, int N) {
    const auto projection_layout{get_tensor_layout(batch_size, N)};
    const Tensor projection{make_identity_tensor(projection_layout.shape())};
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  template <typename LoadType> static constexpr auto get_tiled_copy() {

    // ensures no repeat across the head dimension

    constexpr int elements_per_load{sizeof(LoadType) / sizeof(DType)};
    constexpr int threads_per_row{head_dim / elements_per_load};

    static_assert(
        head_dim % threads_per_row == 0,
        "the head dimension cannot be properly tiled with this thread layout");

    using TPRType = Int<threads_per_row>;
    using EPLType = Int<elements_per_load>;
    constexpr int rows{thread_count / threads_per_row};
    using RowType = Int<rows>;

    static_assert(
        thread_count % threads_per_row == 0,
        "the head dimension cannot be properly tiled with this thread layout");

    static_assert(
        B_r % rows == 0 && B_c % rows == 0,
        "the block size cannot be properly tiled with this thread layout");

    return make_tiled_copy(
        Copy_Atom<UniversalCopy<LoadType>, DType>{},
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

    constexpr int rows{size(get<1>(SourceTensorLayoutType{}))};

    CUTE_UNROLL
    for (int i{0}; i < rows; ++i) {
      auto seq_idx{get<1>(identity_tensor(0, i, 0))};

      if (seq_idx < bound) {
        copy(tiled_copy, source_tensor(_, i, _), destination_tensor(_, i, _));
      } else {
        fill(destination_tensor(_, i, _), fill_value);
      }
    }
  }

  template <bool predicate = false, typename SourceEngineTypeTC,
            typename SourceLayoutTypeTC, typename DestEngineTypeTC,
            typename DestLayoutTypeTC, typename TiledCopyType, typename MMAType,
            typename AEngineTypeMMA, typename ALayoutTypeMMA,
            typename BEngineTypeMMA, typename BLayoutTypeMMA,
            typename CEngineTypeMMA, typename CLayoutTypeMMA,
            typename IdentityEngineType, typename IdentityLayoutType>
  COBRA_S_DEVICE void
  matmul(const Tensor<SourceEngineTypeTC, SourceLayoutTypeTC> &source_slice_cp,
         Tensor<DestEngineTypeTC, DestLayoutTypeTC> &dest_slice_cp,
         const TiledCopyType &tc,
         const Tensor<AEngineTypeMMA, ALayoutTypeMMA> &a_mma_slice,
         const Tensor<BEngineTypeMMA, BLayoutTypeMMA> &b_mma_slice,
         Tensor<CEngineTypeMMA, CLayoutTypeMMA> &c_frag,
         const Tensor<IdentityEngineType, IdentityLayoutType> &b_identity,
         const MMAType &mma, int N) {

    if constexpr (predicate) {
      predicate_copy_tensor(b_identity, source_slice_cp, dest_slice_cp, tc,
                            DType(0), N);
    } else {
      copy(tc, source_slice_cp, dest_slice_cp);
    }
    __syncthreads();

    constexpr size_t mma_m_len{size(get<1>(ALayoutTypeMMA{}))};
    constexpr size_t mma_n_len{size(get<1>(BLayoutTypeMMA{}))};
    constexpr size_t mma_k_len{size(get<2>(BLayoutTypeMMA{}))};

    constexpr size_t elements_per_load{sizeof(VectorizedLoadType) /
                                       sizeof(TensorDType)};

    constexpr size_t slice_factor{mma_m_len};

    constexpr size_t mma_m_size{mma_m_len / slice_factor};

    float4 a_vecs[mma_m_size];
    float4 b_vecs[mma_n_len];

#pragma unroll 8 // too little unrolling hurts ILP to much causes register
                 // spills
    for (size_t k{0}; k < mma_k_len; k += elements_per_load) {

      CUTE_UNROLL
      for (size_t m{0}; m < mma_m_len; m += mma_m_size){

        // 1. Load all A vectors
        CUTE_UNROLL
        for (size_t m_local{0}; m_local < mma_m_size; m_local++) {
          a_vecs[m_local] = *reinterpret_cast<float4 *>(&a_mma_slice(0, m + m_local, k));
        }

        // 2. Load all B vectors
        CUTE_UNROLL
        for (size_t n{0}; n < mma_n_len; n++) {
          b_vecs[n] = *reinterpret_cast<float4 *>(&b_mma_slice(0, n, k));
        }

        // 3. FMAs - all .x first, then .y, then .z, then .w
        CUTE_UNROLL
        for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
          CUTE_UNROLL
          for (size_t n{0}; n < mma_n_len; n++) {
            c_frag(0, m + m_local, n) += a_vecs[m_local].x * b_vecs[n].x;
          }
        }

        CUTE_UNROLL
        for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
          CUTE_UNROLL
          for (size_t n{0}; n < mma_n_len; n++) {
            c_frag(0, m + m_local, n) += a_vecs[m_local].y * b_vecs[n].y;
          }
        }

        CUTE_UNROLL
        for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
          CUTE_UNROLL
          for (size_t n{0}; n < mma_n_len; n++) {
            c_frag(0, m + m_local, n) += a_vecs[m_local].z * b_vecs[n].z;
          }
        }

        CUTE_UNROLL
        for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
          CUTE_UNROLL
          for (size_t n{0}; n < mma_n_len; n++) {
            c_frag(0, m + m_local, n) += a_vecs[m_local].w * b_vecs[n].w;
          }
        }
      }
    }
    // __syncthreads(); // ensure all threads finish reading shared memory before
    //                  // next copy overwrites KV buffer
  }

  template <bool predicate = false, typename MaxTensorEngineType,
            typename RScoresTensorEngineType, typename ProbTensorEngineType,
            typename OutTensorEngineType, typename MaxTensorLayoutType,
            typename ScoresTensorLayoutType, typename ProbTensorLayoutType,
            typename OutTensorLayoutType, typename ScoresIdentityEngineType,
            typename ScoresIdentityLayoutType>
  COBRA_S_DEVICE void update_statistics(
      Tensor<MaxTensorEngineType, MaxTensorLayoutType> &max_tensor,
      Tensor<MaxTensorEngineType, MaxTensorLayoutType> &sum_tensor,
      Tensor<RScoresTensorEngineType, ScoresTensorLayoutType> &r_scores,
      Tensor<ProbTensorEngineType, ProbTensorLayoutType> &prob_tensor,
      Tensor<OutTensorEngineType, OutTensorLayoutType> &out_tensor,
      const Tensor<ScoresIdentityEngineType, ScoresIdentityLayoutType>
          &scores_idty_tensor,
      const DType scale, const int bound) {

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
      auto scores_idty_slice{scores_idty_tensor(_, m, _)};
      auto p_slice{(prob_tensor(_, m, _))};
      auto o_slice{(out_tensor(_, m, _))};

      if constexpr (rank(mma_shape) > 1)
        static_assert(rank(mma_shape) > 1, "not yet implemented");

      auto &current_max{max_tensor(m)};
      auto old_max{current_max};
      auto &current_sum{sum_tensor(m)};

      constexpr size_t slice_size{size(r_score_slice)};

      int adjusted_bound;

      if constexpr (causal_mask) {
        adjusted_bound = get<0>(scores_idty_slice(0)) + 1;
      } else if constexpr (predicate) {
        adjusted_bound = bound;
      } else {
        adjusted_bound = 0;
      }

      CUTE_UNROLL
      for (size_t idx{0}; idx < slice_size; ++idx) {
        // uses hardware unit, removes warp divergence from branch checks
        // each thread may hold multiple values from each row, we find the
        // local maximum first
        if constexpr (predicate || causal_mask) {
          auto n{get<1>(scores_idty_slice(idx))};
          if (n < adjusted_bound) {
            r_score_slice(idx) =
                r_score_slice(idx) * scale; // scale by 1 / sqrt(d)
          } else {
            r_score_slice(idx) = -INFINITY;
          }
        } else {
          r_score_slice(idx) =
              r_score_slice(idx) * scale; // scale by 1 / sqrt(d)
        }

        current_max = cuda::std::max(r_score_slice(idx), current_max);
      }

      current_max = warp_max(current_max);

      // Compute scaling factor for old values
      DType scale_old;
      if constexpr (causal_mask) {
        if (old_max == current_max && current_max == -INFINITY) {
          scale_old = DType(0);
        } else {
          scale_old = expf(old_max - current_max);
        }
      } else {
        scale_old = expf(old_max - current_max);
      }

      // scale the sum
      current_sum = current_sum * scale_old;

      DType local_sum{0};

      // TODO experiment with efficent copies

      CUTE_UNROLL
      for (size_t idx{0}; idx < slice_size; ++idx) {
        auto p_score{r_score_slice(idx)};
        if constexpr (causal_mask) {
          if (old_max == current_max && current_max == -INFINITY) {
            p_score = DType(0);
          } else {
            p_score = expf(p_score - current_max);
          }
        } else {
          p_score = expf(p_score - current_max);
        }

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

    const auto tc_qk{get_tiled_copy<VectorizedLoadType>()};
    const auto tc_v{get_tiled_copy<ScalarLoadType>()};

    const auto tmma{get_tiled_mma()};

    auto kernel_fptr{naive::mha_kernel<Self, decltype(tc_qk), decltype(tc_v),
                                       decltype(tmma)>};

    size_t smem_size{sizeof(SharedStorage)};

    // Set L1 to be SMEM only
    cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaFuncSetAttribute(kernel_fptr,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<grid_dim, block_dim, smem_size>>>(Q, K, V, O, N, scale, tc_qk,
                                                    tc_v, tmma);
  }
};

} // namespace cobraml::kernels