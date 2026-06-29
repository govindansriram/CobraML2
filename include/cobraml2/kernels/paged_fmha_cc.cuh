#pragma once
#include "../algos.cuh"
#include "../macros.cuh"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

// Fused Multi Head Attention, that runs purely on cuda cores
// Based on the Flash Attention 1 algorithm.

namespace cobraml::kernels {

using namespace cute;

template<
  typename PagedEngineType,
  typename PagedLayoutType,
  int warps,
  int head_dim,
  int BKV,
  int page_size
>
struct PagedCopyEngine{
  using VectorizedWidthType = uint128_t;
  using ScalarWidthType = uint32_t;
  static constexpr int width{sizeof(VectorizedWidthType) / sizeof(ScalarWidthType)};
  using ThreadsType = Int<32 * warps>;
  using ThreadsPerRowType = Int<head_dim / width>;

  static_assert(ThreadsType::value % ThreadsPerRowType::value == 0, "Configured thread count is unable to distirbute work evenly");

  using RowsPerThreadBlockType = Int<ThreadsType::value / ThreadsPerRowType::value>;

  static constexpr int iters_per_block{ceil_div(BKV, RowsPerThreadBlockType::value)};

  static_assert(BKV % page_size == 0, "KV Cache Block SIze must be a multiple of KV Cache page size");

  const Tensor<PagedEngineType, PagedLayoutType>& paged_table;
  const int sequence_length;

  COBRA_DEVICE PagedCopyEngine(const Tensor<PagedEngineType, PagedLayoutType> &paged_table, const int sequence_length): 
    paged_table(paged_table), sequence_length(sequence_length){}

  template<
    typename SourceEngineType,
    typename SourceLayout,
    typename DestEngineType,
    typename DestLayout
  >
  COBRA_DEVICE void predicate_paged_copy(
    const Tensor<SourceEngineType, SourceLayout> &src_cache, 
    Tensor<DestEngineType, DestLayout> &dst,
    const int start_row
  ){

    const int thread_row{threadIdx.x / ThreadsPerRowType::value};
    const int thread_col{threadIdx.x % ThreadsPerRowType::value};

    auto coord{make_coord(_, thread_col)};
    auto dst_tiler{make_shape(_1{}, Int<width>{})};

    Tensor dst_iter{
      local_tile(dst, dst_tiler, coord)
    }; // (1, width, block_rows)

    int thread_start_row{start_row + thread_row};

    #pragma unroll
    for (int iter{0}; iter < iters_per_block; ++iter){

      thread_start_row += RowsPerThreadBlockType::value * iter;

      if (thread_start_row < sequence_length){
        int page_table_idx{thread_start_row / page_size};
        int page_idx{paged_table(page_table_idx)};

        Tensor cache_block{src_cache(page_idx, _, _)};

        int page_row{thread_start_row % page_size};

        // this thread's width-wide chunk of the head_dim row
        Tensor src_chunk{local_tile(cache_block(page_row, _),
                                    make_shape(Int<width>{}),
                                    make_coord(thread_col))};

        // vectorized 128-bit copy (== width scalars in one transaction)
        copy(Copy_Atom<UniversalCopy<VectorizedWidthType>, ScalarWidthType>{},
             src_chunk, dst_iter(_, _, thread_start_row - start_row));

      }else if ((thread_start_row - start_row) < BKV){
        fill(dst_iter(_, _, thread_start_row - start_row), 0.0f);
      }

    }

    __syncthreads();

  }

  template<
    typename SourceEngineType,
    typename SourceLayout,
    typename DestEngineType,
    typename DestLayout
  >
  COBRA_DEVICE void paged_copy(
    const Tensor<SourceEngineType, SourceLayout> &src_cache, 
    Tensor<DestEngineType, DestLayout> &dst,
    const int start_row
  ){

    const int thread_row{threadIdx.x / ThreadsPerRowType::value};
    const int thread_col{threadIdx.x % ThreadsPerRowType::value};

    auto coord{make_coord(_, thread_col)};
    auto dst_tiler{make_shape(_1{}, Int<width>{})};

    Tensor dst_iter{
      local_tile(dst, dst_tiler, coord)
    }; // (1, width, block_rows)

    int thread_start_row{start_row + thread_row};

    #pragma unroll
    for (int iter{0}; iter < iters_per_block; ++iter){

      thread_start_row += RowsPerThreadBlockType::value * iter;

      int page_table_idx{thread_start_row / page_size};
      int page_idx{paged_table(page_table_idx)};

      Tensor cache_block{src_cache(page_idx, _, _)};

      int page_row{thread_start_row % page_size};

      // this thread's width-wide chunk of the head_dim row
      Tensor src_chunk{local_tile(cache_block(page_row, _),
                                  make_shape(Int<width>{}),
                                  make_coord(thread_col))};

      // vectorized 128-bit copy (== width scalars in one transaction)
      copy(Copy_Atom<UniversalCopy<VectorizedWidthType>, ScalarWidthType>{},
            src_chunk, dst_iter(_, _, thread_start_row - start_row));

    }
    __syncthreads();

  }
};


// template <bool predicate = false, typename SourceEngineTypeTC,
//           typename SourceLayoutTypeTC, typename DestEngineTypeTC,
//           typename DestLayoutTypeTC, typename TiledCopyType, typename MMAType,
//           typename AEngineTypeMMA, typename ALayoutTypeMMA,
//           typename BEngineTypeMMA, typename BLayoutTypeMMA,
//           typename CEngineTypeMMA, typename CLayoutTypeMMA,
//           typename IdentityEngineType, typename IdentityLayoutType>
// COBRA_DEVICE void
// matmul(const Tensor<SourceEngineTypeTC, SourceLayoutTypeTC> &source_slice_cp,
//         Tensor<DestEngineTypeTC, DestLayoutTypeTC> &dest_slice_cp,
//         const TiledCopyType &tc,
//         const Tensor<AEngineTypeMMA, ALayoutTypeMMA> &a_mma_slice,
//         const Tensor<BEngineTypeMMA, BLayoutTypeMMA> &b_mma_slice,
//         Tensor<CEngineTypeMMA, CLayoutTypeMMA> &c_frag,
//         const Tensor<IdentityEngineType, IdentityLayoutType> &b_identity,
//         const MMAType &mma, int N) {

//   if constexpr (predicate) {
//     predicate_copy_tensor(b_identity, source_slice_cp, dest_slice_cp, tc,
//                           DType(0), N);
//   } else {
//     copy(tc, source_slice_cp, dest_slice_cp);
//   }
//   __syncthreads();

//   constexpr size_t mma_m_len{size(get<1>(ALayoutTypeMMA{}))};
//   constexpr size_t mma_n_len{size(get<1>(BLayoutTypeMMA{}))};
//   constexpr size_t mma_k_len{size(get<2>(BLayoutTypeMMA{}))};

//   constexpr size_t elements_per_load{sizeof(VectorizedLoadType) /
//                                       sizeof(TensorDType)};

//   constexpr size_t slice_factor{mma_m_len};

//   constexpr size_t mma_m_size{mma_m_len / slice_factor};

//   float4 a_vecs[mma_m_size];
//   float4 b_vecs[mma_n_len];

// #pragma unroll 8 // too little unrolling hurts ILP to much causes register
//                 // spills
//   for (size_t k{0}; k < mma_k_len; k += elements_per_load) {

//     CUTE_UNROLL
//     for (size_t m{0}; m < mma_m_len; m += mma_m_size) {

//       // 1. Load all A vectors
//       CUTE_UNROLL
//       for (size_t m_local{0}; m_local < mma_m_size; m_local++) {
//         a_vecs[m_local] =
//             *reinterpret_cast<float4 *>(&a_mma_slice(0, m + m_local, k));
//       }

//       // 2. Load all B vectors
//       CUTE_UNROLL
//       for (size_t n{0}; n < mma_n_len; n++) {
//         b_vecs[n] = *reinterpret_cast<float4 *>(&b_mma_slice(0, n, k));
//       }

//       // 3. FMAs - all .x first, then .y, then .z, then .w
//       CUTE_UNROLL
//       for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
//         CUTE_UNROLL
//         for (size_t n{0}; n < mma_n_len; n++) {
//           c_frag(0, m + m_local, n) += a_vecs[m_local].x * b_vecs[n].x;
//         }
//       }

//       CUTE_UNROLL
//       for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
//         CUTE_UNROLL
//         for (size_t n{0}; n < mma_n_len; n++) {
//           c_frag(0, m + m_local, n) += a_vecs[m_local].y * b_vecs[n].y;
//         }
//       }

//       CUTE_UNROLL
//       for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
//         CUTE_UNROLL
//         for (size_t n{0}; n < mma_n_len; n++) {
//           c_frag(0, m + m_local, n) += a_vecs[m_local].z * b_vecs[n].z;
//         }
//       }

//       CUTE_UNROLL
//       for (size_t m_local{0}; m_local < mma_m_size; ++m_local) {
//         CUTE_UNROLL
//         for (size_t n{0}; n < mma_n_len; n++) {
//           c_frag(0, m + m_local, n) += a_vecs[m_local].w * b_vecs[n].w;
//         }
//       }
//     }
//   }
// }

// template <bool predicate = false, typename MaxTensorEngineType,
//           typename RScoresTensorEngineType, typename ProbTensorEngineType,
//           typename OutTensorEngineType, typename MaxTensorLayoutType,
//           typename ScoresTensorLayoutType, typename ProbTensorLayoutType,
//           typename OutTensorLayoutType, typename ScoresIdentityEngineType,
//           typename ScoresIdentityLayoutType>
// COBRA_DEVICE void update_statistics(
//     Tensor<MaxTensorEngineType, MaxTensorLayoutType> &max_tensor,
//     Tensor<MaxTensorEngineType, MaxTensorLayoutType> &sum_tensor,
//     Tensor<RScoresTensorEngineType, ScoresTensorLayoutType> &r_scores,
//     Tensor<ProbTensorEngineType, ProbTensorLayoutType> &prob_tensor,
//     Tensor<OutTensorEngineType, OutTensorLayoutType> &out_tensor,
//     const Tensor<ScoresIdentityEngineType, ScoresIdentityLayoutType>
//         &scores_idty_tensor,
//     const DType scale, const int bound, const int start_pos = 0) {

//   static_assert(rank_v<ScoresTensorLayoutType> == 3,
//                 "Per Register Attention scores must be 3 dimensional (mma, "
//                 "mma_m, mma_n)");

//   static_assert(rank_v<MaxTensorLayoutType> == 1,
//                 "Per register, row maxes, muse be 1 dimensional");

//   using MMAShape = decltype(get<0>(ScoresTensorLayoutType{}));
//   constexpr size_t mma_m{size(get<1>(ScoresTensorLayoutType{}))};

//   static_assert(rank(MMAShape{}) == 1, "not yet implemented");

//   CUTE_UNROLL
//   for (size_t m{0}; m < mma_m; ++m) {

//     auto r_score_slice{r_scores(_, m, _)};
//     auto scores_idty_slice{scores_idty_tensor(_, m, _)};
//     auto p_slice{(prob_tensor(_, m, _))};
//     auto o_slice{(out_tensor(_, m, _))};

//     auto &current_max{max_tensor(m)};
//     auto old_max{current_max};
//     auto &current_sum{sum_tensor(m)};

//     constexpr size_t slice_size{size(r_score_slice)};

//     int adjusted_bound;

//     if constexpr (causal_mask) {
//       adjusted_bound = get<0>(scores_idty_slice(0)) + start_pos + 1;
//     } else if constexpr (predicate) {
//       adjusted_bound = bound;
//     } else {
//       adjusted_bound = 0;
//     }

//     CUTE_UNROLL
//     for (size_t idx{0}; idx < slice_size; ++idx) {
//       // uses hardware unit, removes warp divergence from branch checks
//       // each thread may hold multiple values from each row, we find the
//       // local maximum first
//       if constexpr (predicate || causal_mask) {
//         auto n{get<1>(scores_idty_slice(idx))};
//         if (n < adjusted_bound) {
//           r_score_slice(idx) =
//               r_score_slice(idx) * scale; // scale by 1 / sqrt(d)
//         } else {
//           r_score_slice(idx) = -INFINITY;
//         }
//       } else {
//         r_score_slice(idx) =
//             r_score_slice(idx) * scale; // scale by 1 / sqrt(d)
//       }

//       current_max = cuda::std::max(r_score_slice(idx), current_max);
//     }

//     current_max = warp_max(current_max);

//     // Compute scaling factor for old values
//     DType scale_old;
//     if constexpr (causal_mask) {
//       if (old_max == current_max && current_max == -INFINITY) {
//         scale_old = DType(0);
//       } else {
//         scale_old = expf(old_max - current_max);
//       }
//     } else {
//       scale_old = expf(old_max - current_max);
//     }

//     // scale the sum
//     current_sum = current_sum * scale_old;

//     DType local_sum{0};

//     // TODO experiment with efficent copies

//     CUTE_UNROLL
//     for (size_t idx{0}; idx < slice_size; ++idx) {
//       auto p_score{r_score_slice(idx)};
//       if constexpr (causal_mask) {
//         if (old_max == current_max && current_max == -INFINITY) {
//           p_score = DType(0);
//         } else {
//           p_score = expf(p_score - current_max);
//         }
//       } else {
//         p_score = expf(p_score - current_max);
//       }

//       local_sum += p_score;
//       // write to probs tensor
//       p_slice(idx) = p_score;
//       // reset registers
//       r_score_slice(idx) = 0;
//     }

//     current_sum += warp_sum(local_sum);

//     CUTE_UNROLL
//     for (size_t i{0}; i < size(o_slice); i++) {
//       o_slice(i) *= scale_old;
//     }
//   }
// }

// grid(num_heads, num_requests, ceil(max_seq_len_q / B_r))

/**
 * q: packed query tokens (all requests in the batch have their query tokens packed 
 * into a continuous buffer) (N, num_heads, head_dim) 
 * 
 * k_cache, the k pool (num_pages, page_size, num_heads, head_dim)
 * v_cache, the v pool (num_pages, page_size, num_heads, head_dim)
 * 
 * KV pool shape (2, num_layers, num_pages, page_size, num_heads, head_dim)
 * 
 * o, the output tensor same size as q (N, num_heads, D)
 * 
 * page_table: A table of size (max_running_req + 1, ceil_div(max_seq_len, block_size)). 
 * This is created once upon engine creation, each element details the start index of that 
 * block.
 * 
 * cache_seqlen: How many KV tokens are used for a request [num_requests]
 * 
 * cu_seqlens_q: The cumulative sum of q tokens per request detailing the start and stop index of the q segment
 * [num_requests + 1] -> [0, 4, 7, 11]
 * 
 * cu_seqlens_k_new: The cumulative sequence lengths of all new kv tokens per request 
 * cache_seqlen: [8, 2, 4, 4], cu_seqlens_k_new: [0, 8, 9, 10, 11] means request 1 has 8 new tokens, the rest have 
 * 1. This info is needed for causal masking
 */

template<
  int BQ,
  int BKV,
  int head_dim
>
struct SharedStorage {
  using BQType = Int<BQ>;
  using BKVType = Int<BKV>;
  using HeadDimType = Int<head_dim>;

  ArrayEngine<float, BQ * head_dim> block_q;
  ArrayEngine<float, BKV * head_dim> block_KV;
  ArrayEngine<float, BQ * BKV> block_p;

  using QLayoutType = decltype(make_layout(make_shape(BQType{}, HeadDimType{}), LayoutRight{}));
  using KVLayoutType = decltype(make_layout(make_shape(BKVType{}, HeadDimType{}), LayoutRight{}));
  using VTLayoutType = decltype(make_layout(make_shape(HeadDimType{}, BKVType{}), LayoutRight{}));
  using PLayoutType = decltype(make_layout(make_shape(BQType{}, BKVType{}), LayoutRight{}));
};


template <
    typename FloatEngineType, // global pointer engine for q/k/v/o (float)
    typename IntEngineType,    // global pointer engine for index/count tensors (int)
    typename QOLayoutType,
    typename KVLayoutType,
    typename PageTableLayoutType,
    typename CacheSeqlenLayoutType,
    typename CuSeqlenQLayoutType,
    typename CuSeqlenKNewLayoutType,
    typename TiledMMAType,
    bool causal,
    int BQ,
    int BKV
>
__global__ void paged_mha_cc_kernel(
    const Tensor<FloatEngineType, QOLayoutType> q,
    const Tensor<FloatEngineType, KVLayoutType> k_cache,
    const Tensor<FloatEngineType, KVLayoutType> v_cache,
    Tensor<FloatEngineType, QOLayoutType> o,
    const Tensor<IntEngineType, PageTableLayoutType> page_table,
    const Tensor<IntEngineType, CacheSeqlenLayoutType> cache_seqlen,
    const Tensor<IntEngineType, CuSeqlenQLayoutType> cu_seqlens_q,
    const Tensor<IntEngineType, CuSeqlenKNewLayoutType> cu_seqlens_k_new,
    const float scale,
    TiledMMAType t_mma) {

  uint32_t head{blockIdx.x};
  uint32_t request{blockIdx.y};
  uint32_t seq_start{blockIdx.z * BQ};

  int q_start{cu_seqlens_q[request]};
  int q_end{cu_seqlens_q[request + 1]};
  int N{q_end - q_start}; // sequence length
  int N_KV{cache_seqlen[request]};

  if (thread0()){
    print(N); print("\n");
    print_tensor(cu_seqlens_q);
  }

  constexpr int head_dim{shape<3>(KVLayoutType{})};

  if (q_start >= N)
    return;

  Tensor q_slice{
    make_tensor(
      &q(q_start, head, _0{}),
      make_layout(make_shape(N, shape<2>(q)), LayoutRight{}) 
    )
  };

  Tensor o_slice{
    make_tensor(
      &o(q_start, head, _0{}),
      make_layout(make_shape(N, shape<2>(o)), LayoutRight{}) 
    )
  };

  Tensor k_cache_view{k_cache(_, _, head, _)};
  Tensor v_cache_view{k_cache(_, _, head, _)};
  Tensor pages{page_table(request, _)};

  constexpr Int<head_dim> d{};

  extern __shared__ char shared_memory[];
  using SharedStorageType = SharedStorage<BQ, BKV, d>;
  SharedStorageType *shared_storage{
      reinterpret_cast<SharedStorageType *>(shared_memory)};

  Tensor shared_q{make_tensor(make_smem_ptr(shared_storage->block_q.begin()),
                              typename SharedStorageType::QLayoutType{})};

  Tensor shared_k{make_tensor(make_smem_ptr(shared_storage->block_KV.begin()),
                              typename SharedStorageType::KVLayoutType{})};

  Tensor shared_v{make_tensor(make_smem_ptr(shared_storage->block_KV.begin()),
                              typename SharedStorageType::KVLayoutType{})};

  Tensor trans_shared_v{
      make_tensor(make_smem_ptr(shared_storage->block_KV.begin()),
                  typename SharedStorageType::VTLayoutType{})};

  Tensor shared_p{make_tensor(make_smem_ptr(shared_storage->block_p.begin()),
                              typename SharedStorageType::PLayoutType{})};

  // // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

  constexpr Int<BQ> bq{};
  constexpr Int<BKV> bkv{};

  auto qo_coord{make_coord(blockIdx.z, _0{})};
  // auto kv_coord{make_coord(_, 0)};

  auto qo_tiler{make_shape(Int<BQ>{}, d)};
  auto scores_tiler{make_shape(Int<BQ>{}, Int<BKV>{})};

  Tensor q_iterator{local_tile(q_slice, qo_tiler, qo_coord)}; // (B_r, d)
  Tensor o_iterator{local_tile(o_slice, qo_tiler, qo_coord)}; // (B_r, d)

  int iters{ceil_div(N_KV, BKV)}; // N_kv

  // // t prefix means unique to this thread
  // ThrCopy thr_copy_qk{tc_qk.get_slice(threadIdx.x)};
  // ThrCopy thr_copy_v{tc_v.get_slice(threadIdx.x)};

  // const Tensor tQ_global_part{thr_copy_qk.partition_S(q_iterator)};
  // Tensor tQ_shared_part{thr_copy_qk.partition_D(shared_q)};

  // const Tensor tK_global_part_iter{thr_copy_qk.partition_S(k_iterator)};
  // Tensor tK_shared_part{thr_copy_qk.partition_D(shared_k)};

  // const Tensor tV_global_part_iter{thr_copy_v.partition_S(v_iterator)};
  // Tensor tV_shared_part{thr_copy_v.partition_D(shared_v)};

  ThrMMA thr_mma_qk{t_mma.get_slice(threadIdx.x)};

  Tensor q_mma{thr_mma_qk.partition_A(shared_q)};
  Tensor k_mma{thr_mma_qk.partition_B(shared_k)};
  Tensor p_mma{thr_mma_qk.partition_C(shared_p)};

  Tensor p_mma2{thr_mma_qk.partition_A(shared_p)};
  Tensor v_mma{thr_mma_qk.partition_B(trans_shared_v)};
  Tensor g_out_mma{thr_mma_qk.partition_C(o_iterator)};
  Tensor r_out_mma{thr_mma_qk.make_fragment_C(g_out_mma)};

  // auto q_head_idty{MHAType::identity_slice_head(batch_size, N_q)};
  // auto kv_head_idty{MHAType::identity_slice_head(batch_size, N_kv)};

  // // make q identity tensor
  // auto q_head_slice_idty{local_tile(q_head_idty, q_tiler, q_coord)}; // (B_r, d)
  // auto tQ_idty_part{thr_copy_qk.partition_S(q_head_slice_idty)};

  // // make k identity tensor
  // auto kv_iterator_idty{local_tile(kv_head_idty, kv_tiler,
  //                                  kv_coord)}; // (B_c, d, ceil(N_kv / B_c))
  // auto k_idty_part{thr_copy_qk.partition_S(kv_iterator_idty)};

  // // make v identity tensor
  // auto v_idty_part{thr_copy_v.partition_S(kv_iterator_idty)};

  // scores identity tensor
  auto scores_idty{make_identity_tensor(make_shape(N, N_KV))};
  auto scores_tile_idty{
      local_tile(scores_idty, scores_tiler,
                 make_coord(blockIdx.z, _))}; // (B_r, B_c, ceil(N / B_c))
  Tensor scores_slice_idty{thr_mma_qk.partition_C(scores_tile_idty)};

  // // out identity tensor
  // auto o_iterator_idty{local_tile(q_head_idty, q_tiler, q_coord)}; // (B_r, d)
  // Tensor o_mma_idty{thr_mma_qk.partition_C(o_iterator_idty)};

  // predicated copy
  // MHAType::predicate_copy_tensor(tQ_idty_part, tQ_global_part, tQ_shared_part,
  //                                tc_qk, DType(0), N_q);

  // auto mma_m{select<1>(q_mma.shape())};

  // Tensor r_scores_mma{thr_mma_qk.make_fragment_C(p_mma)};
  // clear(r_scores_mma); // Zero the accumulator

  // // start with the lowest possible value
  // auto m{make_tensor<DType>(mma_m)};
  // auto l{make_tensor<DType>(mma_m)};
  // fill(m, -INFINITY);
  // clear(l); // zero the sums

  // // Do the block that needs predication first

  // MHAType::matmul<true>(tK_global_part_iter(_, _, _, iters - 1), tK_shared_part,
  //                       tc_qk, q_mma, k_mma, r_scores_mma,
  //                       k_idty_part(_, _, _, iters - 1), t_mma, N_kv);

  // MHAType::update_statistics<true>(m, l, r_scores_mma, p_mma, r_out_mma,
  //                                  scores_slice_idty(_, _, _, iters - 1), scale,
  //                                  N_kv, start_pos);

  // __syncthreads(); // ensure all K reads done before V overwrites KV buffer
  // MHAType::matmul<true>(tV_global_part_iter(_, _, _, iters - 1), tV_shared_part,
  //                       tc_v, p_mma2, v_mma, r_out_mma,
  //                       v_idty_part(_, _, _, iters - 1), t_mma, N_kv);

  // // do the rest of blocks that don't need predication
  // for (int iter{static_cast<int>(iters) - 2}; iter > -1; --iter) {
  //   __syncthreads();
  //   MHAType::matmul(tK_global_part_iter(_, _, _, iter), tK_shared_part, tc_qk,
  //                   q_mma, k_mma, r_scores_mma, k_idty_part(_, _, _, iter),
  //                   t_mma, N_kv);

  //   MHAType::update_statistics(m, l, r_scores_mma, p_mma, r_out_mma,
  //                              scores_slice_idty(_, _, _, iter), scale, N_kv,
  //                              start_pos);

  //   __syncthreads(); // ensure all K reads done before V overwrites KV buffer
  //   MHAType::matmul(tV_global_part_iter(_, _, _, iter), tV_shared_part, tc_v,
  //                   p_mma2, v_mma, r_out_mma, v_idty_part(_, _, _, iter), t_mma,
  //                   N_kv);
  // }

  // auto mma_shape{get<0>(r_out_mma.layout())};
  // auto m_rows{size(get<1>(r_out_mma.layout()))};

  // static_assert(rank(mma_shape) == 1,
  //               "only rank 1 mma shape is currently supported");

  // CUTE_UNROLL
  // for (size_t m_row{0}; m_row < m_rows; ++m_row) {
  //   auto out_slice{r_out_mma(_, m_row, _)};

  //   CUTE_UNROLL
  //   for (size_t idx{0}; idx < size(out_slice); ++idx) {
  //     out_slice(idx) = out_slice(idx) / l(m_row);
  //   }
  // }

  // constexpr int write_rows{size(get<1>(g_out_mma.shape()))};

  // CUTE_UNROLL
  // for (size_t i{0}; i < write_rows; ++i) {
  //   auto seq_idx{get<1>(o_mma_idty(0, i, 0))};

  //   if (seq_idx < N_q)
  //     copy(r_out_mma(_, i, _), g_out_mma(_, i, _));
  // }
}

template<
  int BQ = 16,
  int BKV = 16,
  int warps_per_block = 4,
  bool causal_mask = false
>
struct PagedFMHACC_Config {};

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
template <
          typename FloatEngineType,
          typename IntEngineType,
          typename KVLayoutType,
          typename PagedTableLayout,
          int BQ,
          int BKV,
          int warps_per_block,
          bool causal_mask
        >
struct PagedFMHACC {

  const Tensor<FloatEngineType, KVLayoutType> Kcache;
  const Tensor<FloatEngineType, KVLayoutType> Vcache;
  const Tensor<IntEngineType, PagedTableLayout> paged_table;
  const PagedFMHACC_Config<BQ, BKV, warps_per_block, causal_mask> config{};

  static_assert(rank(KVLayoutType{}) == 4, "KV Cache must have 4 modes");

  PagedFMHACC(
    const Tensor<FloatEngineType, KVLayoutType> Kcache,
    const Tensor<FloatEngineType, KVLayoutType> Vcache,
    const Tensor<IntEngineType, PagedTableLayout> paged_table,
    const PagedFMHACC_Config<BQ, BKV, warps_per_block, causal_mask> config
  ): Kcache(Kcache), Vcache(Vcache), paged_table(paged_table), config(config){}

  using TensorDType = float;

  using NumHeadsType = decltype(shape<2>(KVLayoutType{}));
  using HeadDimType = decltype(shape<3>(KVLayoutType{}));
  using BPagedType = decltype(shape<1>(KVLayoutType{}));

  using BQType = Int<BQ>;
  using BKVType = Int<BKV>;

  static_assert(BKV % BPagedType::value == 0, "The KV Block Size must be a multiple of the Paged Block Size");
  static constexpr int threads_per_block{warps_per_block * 32};

  static constexpr auto get_tiled_mma() {

    static_assert(threads_per_block % 32 == 0,
                  "thread_count must be a multiple of warp_size");

    using RowType = Int<threads_per_block / 32>;

    auto t_mma{
        make_tiled_mma(UniversalFMA<TensorDType, TensorDType, TensorDType>{},
                       Layout<Shape<RowType, _32>,
                              Stride<_32, _1>>{})}; // 16x16x1 UniversalFMA

    return t_mma;
  }

  template<
    typename QOLayoutType,
    typename CacheSeqlenLayoutType,
    typename CuSeqlenQLayoutType,
    typename CuSeqlenKNewLayoutType
  >
  void operator()(
    const Tensor<FloatEngineType, QOLayoutType> q,
    Tensor<FloatEngineType, QOLayoutType> o,
    const Tensor<IntEngineType, CacheSeqlenLayoutType> cache_seqlen,
    const Tensor<IntEngineType, CuSeqlenQLayoutType> cu_seqlens_q,
    const Tensor<IntEngineType, CuSeqlenKNewLayoutType> cu_seqlens_k_new,
    const int num_requests,
    const int max_seq_len_q) {

    dim3 grid_dim{NumHeadsType::value, static_cast<uint32_t>(num_requests), static_cast<uint32_t>(ceil_div(max_seq_len_q, BQ))};
    dim3 block_dim{threads_per_block};

    const auto tmma{get_tiled_mma()};

    const auto scale{rsqrt(static_cast<TensorDType>(HeadDimType::value))};

    auto kernel_fptr{paged_mha_cc_kernel<
        FloatEngineType, IntEngineType, QOLayoutType, KVLayoutType,
        PagedTableLayout, CacheSeqlenLayoutType, CuSeqlenQLayoutType,
        CuSeqlenKNewLayoutType, decltype(tmma), causal_mask, BQ, BKV>};

    size_t smem_size{sizeof(SharedStorage<BQ, BKV, HeadDimType::value>)};

    cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaFuncSetAttribute(kernel_fptr,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<grid_dim, block_dim, smem_size>>>(
        q, Kcache, Vcache, o, paged_table, cache_seqlen, cu_seqlens_q, cu_seqlens_k_new, scale, tmma);
  }
};

} // namespace cobraml::kernels
