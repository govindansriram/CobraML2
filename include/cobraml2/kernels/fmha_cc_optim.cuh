#pragma once
#include "../algos.cuh"
#include "../macros.cuh"
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>


namespace cobraml::kernels {

using namespace cute;

namespace pipelined {

/**
 * @brief pipelined gpu mha (cp.async + multi-stage shared memory)
 */
template <typename MHAType, typename TiledCopyTypeQK, typename TiledCopyTypeV,
          typename TiledMMAType>
__global__ void mha_kernel(const typename MHAType::TensorDType *__restrict__ Q,
                           const typename MHAType::TensorDType *__restrict__ K,
                           const typename MHAType::TensorDType *__restrict__ V,
                           typename MHAType::TensorDType *__restrict__ O,
                           const int N_q, const int N_kv, const int start_pos,
                           const typename MHAType::TensorDType scale,
                           TiledCopyTypeQK tc_qk, TiledCopyTypeV tc_v,
                           TiledMMAType t_mma) {

  using DType = typename MHAType::TensorDType;
  using SharedStorageType = typename MHAType::SharedStorage;
  constexpr int kStages = SharedStorageType::kStages;

  size_t batch_size{gridDim.y};

  Tensor q_head{MHAType::slice_head(Q, batch_size, N_q)};
  const Tensor k_head{MHAType::slice_head(K, batch_size, N_kv)};
  const Tensor v_head{MHAType::slice_head(V, batch_size, N_kv)};
  Tensor o_head{MHAType::template slice_head<true>(O, batch_size, N_q)};

  extern __shared__ char shared_memory[];
  SharedStorageType *shared_storage{
      reinterpret_cast<SharedStorageType *>(shared_memory)};

  Tensor shared_q{make_tensor(make_smem_ptr(shared_storage->Q.begin()),
                              typename SharedStorageType::QLayoutType{})};
  Tensor shared_p{make_tensor(make_smem_ptr(shared_storage->P.begin()),
                              typename SharedStorageType::PLayoutType{})};

  constexpr int kStageElemsKV = SharedStorageType::kStageElemsKV;

  constexpr typename MHAType::HeadDimType d{};
  constexpr typename MHAType::QueryRowsType B_r{};
  constexpr typename MHAType::KVColsType B_c{};

  auto q_coord{make_coord(blockIdx.z, 0)};
  auto kv_coord{make_coord(_, 0)};

  auto q_tiler{make_shape(B_r, d)};
  auto kv_tiler{make_shape(B_c, d)};
  auto scores_tiler{make_shape(B_r, B_c)};

  Tensor q_iterator{local_tile(q_head, q_tiler, q_coord)};
  Tensor k_iterator{local_tile(k_head, kv_tiler, kv_coord)};
  Tensor v_iterator{local_tile(v_head, kv_tiler, kv_coord)};
  Tensor o_iterator{local_tile(o_head, q_tiler, q_coord)};

  const int iters = static_cast<int>(size<2>(k_iterator));

  ThrCopy thr_copy_qk{tc_qk.get_slice(threadIdx.x)};
  ThrCopy thr_copy_v{tc_v.get_slice(threadIdx.x)};

  const Tensor tQ_global_part{thr_copy_qk.partition_S(q_iterator)};
  Tensor tQ_shared_part{thr_copy_qk.partition_D(shared_q)};

  const Tensor tK_global_part_iter{thr_copy_qk.partition_S(k_iterator)};
  const Tensor tV_global_part_iter{thr_copy_v.partition_S(v_iterator)};

  ThrMMA thr_mma_qk{t_mma.get_slice(threadIdx.x)};

  Tensor q_mma{thr_mma_qk.partition_A(shared_q)};
  Tensor p_mma{thr_mma_qk.partition_C(shared_p)};
  Tensor p_mma2{thr_mma_qk.partition_A(shared_p)};
  Tensor g_out_mma{thr_mma_qk.partition_C(o_iterator)};
  Tensor r_out_mma{thr_mma_qk.make_fragment_C(g_out_mma)};

  auto q_head_idty{MHAType::identity_slice_head(batch_size, N_q)};
  auto kv_head_idty{MHAType::identity_slice_head(batch_size, N_kv)};

  auto q_head_slice_idty{local_tile(q_head_idty, q_tiler, q_coord)};
  auto tQ_idty_part{thr_copy_qk.partition_S(q_head_slice_idty)};

  auto kv_iterator_idty{local_tile(kv_head_idty, kv_tiler, kv_coord)};
  auto k_idty_part{thr_copy_qk.partition_S(kv_iterator_idty)};
  auto v_idty_part{thr_copy_v.partition_S(kv_iterator_idty)};

  auto scores_idty{make_identity_tensor(make_shape(N_q, N_kv))};
  auto scores_tile_idty{
      local_tile(scores_idty, scores_tiler, make_coord(blockIdx.z, _))};
  Tensor scores_slice_idty{thr_mma_qk.partition_C(scores_tile_idty)};

  auto o_iterator_idty{local_tile(q_head_idty, q_tiler, q_coord)};
  Tensor o_mma_idty{thr_mma_qk.partition_C(o_iterator_idty)};

  // Q is loaded once and reused across all iters; K and V are multi-staged.
  MHAType::predicate_copy_tensor(tQ_idty_part, tQ_global_part, tQ_shared_part,
                                 tc_qk, DType(0), N_q);
  cp_async_fence();

  auto mma_m{select<1>(q_mma.shape())};
  Tensor r_scores_mma{thr_mma_qk.make_fragment_C(p_mma)};
  clear(r_scores_mma);

  auto m{make_tensor<DType>(mma_m)};
  auto l{make_tensor<DType>(mma_m)};
  fill(m, -INFINITY);
  clear(l);

  // Prefetch kStages of K and V. Empty fences when iter_idx < 0 keep
  // wait_group accounting uniform at 2*kStages + 1 commits in flight.
  CUTE_UNROLL
  for (int s = 0; s < kStages; ++s) {
    const int iter_idx = iters - 1 - s;
    if (iter_idx >= 0) {
      const bool needs_pred = (iter_idx == iters - 1);
      auto shared_k_s = make_tensor(
          make_smem_ptr(shared_storage->K.begin() + s * kStageElemsKV),
          typename SharedStorageType::KLayoutType{});
      auto tK_shared_s = thr_copy_qk.partition_D(shared_k_s);
      if (needs_pred) {
        MHAType::predicate_copy_tensor(
            k_idty_part(_, _, _, iter_idx),
            tK_global_part_iter(_, _, _, iter_idx), tK_shared_s, tc_qk,
            DType(0), N_kv);
      } else {
        copy(tc_qk, tK_global_part_iter(_, _, _, iter_idx), tK_shared_s);
      }
    }
    cp_async_fence();

    if (iter_idx >= 0) {
      const bool needs_pred = (iter_idx == iters - 1);
      auto shared_v_s = make_tensor(
          make_smem_ptr(shared_storage->V.begin() + s * kStageElemsKV),
          typename SharedStorageType::VLayoutType{});
      auto tV_shared_s = thr_copy_v.partition_D(shared_v_s);
      if (needs_pred) {
        MHAType::predicate_copy_tensor(
            v_idty_part(_, _, _, iter_idx),
            tV_global_part_iter(_, _, _, iter_idx), tV_shared_s, tc_v,
            DType(0), N_kv);
      } else {
        copy(tc_v, tV_global_part_iter(_, _, _, iter_idx), tV_shared_s);
      }
    }
    cp_async_fence();
  }

  // Boundary iter peeled so update_statistics<true> is a compile-time choice.
  if (iters > 0) {
    constexpr int s_first = 0;
    const int prefetch_iter = iters - 1 - kStages;

    // wait<2*kStages-1> drains the 2 oldest commits: Q and K[0].
    cp_async_wait<2 * kStages - 1>();
    __syncthreads();

    {
      auto shared_k_s = make_tensor(
          make_smem_ptr(shared_storage->K.begin() + s_first * kStageElemsKV),
          typename SharedStorageType::KLayoutType{});
      auto k_mma_s = thr_mma_qk.partition_B(shared_k_s);
      MHAType::mma_fma_inner(q_mma, k_mma_s, r_scores_mma);
    }

    __syncthreads(); // K[0] consumed; safe to overwrite via prefetch below

    if (prefetch_iter >= 0) {
      auto shared_k_s = make_tensor(
          make_smem_ptr(shared_storage->K.begin() + s_first * kStageElemsKV),
          typename SharedStorageType::KLayoutType{});
      auto tK_shared_s = thr_copy_qk.partition_D(shared_k_s);
      copy(tc_qk, tK_global_part_iter(_, _, _, prefetch_iter), tK_shared_s);
    }
    cp_async_fence();

    MHAType::template update_statistics<true>(
        m, l, r_scores_mma, p_mma, r_out_mma,
        scores_slice_idty(_, _, _, iters - 1), scale, N_kv, start_pos);

    cp_async_wait<2 * kStages - 1>();
    __syncthreads(); // V[0] and shared_p both visible after this

    {
      auto trans_shared_v_s = make_tensor(
          make_smem_ptr(shared_storage->V.begin() + s_first * kStageElemsKV),
          typename SharedStorageType::VTransposedLayoutType{});
      auto v_mma_s = thr_mma_qk.partition_B(trans_shared_v_s);
      MHAType::mma_fma_inner(p_mma2, v_mma_s, r_out_mma);
    }

    __syncthreads(); // V[0] consumed; safe to overwrite via prefetch below

    if (prefetch_iter >= 0) {
      auto shared_v_s = make_tensor(
          make_smem_ptr(shared_storage->V.begin() + s_first * kStageElemsKV),
          typename SharedStorageType::VLayoutType{});
      auto tV_shared_s = thr_copy_v.partition_D(shared_v_s);
      copy(tc_v, tV_global_part_iter(_, _, _, prefetch_iter), tV_shared_s);
    }
    cp_async_fence();
  }

  for (int iter_idx = iters - 2; iter_idx >= 0; --iter_idx) {
    const int s = (iters - 1 - iter_idx) % kStages;
    const int prefetch_iter = iter_idx - kStages;

    cp_async_wait<2 * kStages - 1>();
    __syncthreads();

    {
      auto shared_k_s = make_tensor(
          make_smem_ptr(shared_storage->K.begin() + s * kStageElemsKV),
          typename SharedStorageType::KLayoutType{});
      auto k_mma_s = thr_mma_qk.partition_B(shared_k_s);
      MHAType::mma_fma_inner(q_mma, k_mma_s, r_scores_mma);
    }

    __syncthreads();

    if (prefetch_iter >= 0) {
      auto shared_k_s = make_tensor(
          make_smem_ptr(shared_storage->K.begin() + s * kStageElemsKV),
          typename SharedStorageType::KLayoutType{});
      auto tK_shared_s = thr_copy_qk.partition_D(shared_k_s);
      copy(tc_qk, tK_global_part_iter(_, _, _, prefetch_iter), tK_shared_s);
    }
    cp_async_fence();

    MHAType::update_statistics(m, l, r_scores_mma, p_mma, r_out_mma,
                               scores_slice_idty(_, _, _, iter_idx), scale,
                               N_kv, start_pos);

    cp_async_wait<2 * kStages - 1>();
    __syncthreads();

    {
      auto trans_shared_v_s = make_tensor(
          make_smem_ptr(shared_storage->V.begin() + s * kStageElemsKV),
          typename SharedStorageType::VTransposedLayoutType{});
      auto v_mma_s = thr_mma_qk.partition_B(trans_shared_v_s);
      MHAType::mma_fma_inner(p_mma2, v_mma_s, r_out_mma);
    }

    __syncthreads();

    if (prefetch_iter >= 0) {
      auto shared_v_s = make_tensor(
          make_smem_ptr(shared_storage->V.begin() + s * kStageElemsKV),
          typename SharedStorageType::VLayoutType{});
      auto tV_shared_s = thr_copy_v.partition_D(shared_v_s);
      copy(tc_v, tV_global_part_iter(_, _, _, prefetch_iter), tV_shared_s);
    }
    cp_async_fence();
  }

  cp_async_wait<0>();

  // Normalize r_out by row sums and write back to global.
  auto mma_shape{get<0>(r_out_mma.layout())};
  auto m_rows{size(get<1>(r_out_mma.layout()))};

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

    if (seq_idx < N_q)
      copy(r_out_mma(_, i, _), g_out_mma(_, i, _));
  }
}
} // namespace pipelined

/**
 * @brief
 *
 * @tparam head_count
 * @tparam head_dim the length of each head
 * @tparam B_r how many sequences of q to process at once
 * @tparam B_c how many sequences of K and V to process at once
 * @tparam DType
 * @tparam thread_count
 * @tparam kStages_ depth of the shared-memory pipeline (>=1)
 */
template <int head_count, int head_dim, int B_r, int B_c, typename DType,
          int thread_count = 128, bool causal_mask = false,
          bool qkv_contigous_buffer = false, int kStages_ = 2>
struct FMHA {

  using TensorDType = DType;
  using Self = FMHA<head_count, head_dim, B_r, B_c, DType, thread_count,
                    causal_mask, qkv_contigous_buffer, kStages_>;

  using NumHeadsType = Int<head_count>;
  using HeadDimType = Int<head_dim>;
  using QueryRowsType = Int<B_r>;
  using KVColsType = Int<B_c>;

  using VectorizedLoadType = uint128_t;
  using ScalarLoadType = uint32_t;

  static constexpr int kStages = kStages_;

  struct SharedStorage {
    static constexpr int kStages = kStages_;
    static constexpr int kStageElemsKV = B_c * head_dim;

    // Swizzle atom dimensions
    static constexpr int kSwizzleAtomRows = 8;
    static constexpr int kSwizzleAtomCols = 32;

    static_assert(
        B_r % kSwizzleAtomRows == 0,
        "B_r must be divisible by 8 (swizzle atom row size) for Q layout");
    static_assert(
        B_c % kSwizzleAtomCols == 0,
        "B_c must be divisible by 32 (swizzle atom col size) for V layout");
    static_assert(head_dim % kSwizzleAtomCols == 0,
                  "head_dim must be divisible by 32 (swizzle atom col size) "
                  "for Q/K layouts");
    static_assert(kStages >= 1 && kStages <= 4,
                  "kStages must be in [1, 4]");

    ArrayEngine<DType, B_r * head_dim> Q;
    ArrayEngine<DType, kStages * B_c * head_dim> K;
    ArrayEngine<DType, kStages * B_c * head_dim> V;
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

  template <bool is_o = false>
  COBRA_S_DEVICE auto get_tensor_layout(size_t batch_size, size_t N) {

    if constexpr (qkv_contigous_buffer && !is_o) {
      Int<head_dim * head_count * 3> embed3{};
      return make_layout(
          make_shape(batch_size, N, NumHeadsType{}, HeadDimType{}),
          make_stride(embed3.value * N, embed3, HeadDimType{}, _1{}));
    } else {
      return make_layout(
          make_shape(batch_size, N, NumHeadsType{}, HeadDimType{}),
          LayoutRight{});
    }
  }

  template <bool is_o = false, typename PtrType>
  COBRA_S_DEVICE auto slice_head(PtrType g_ptr, int batch_size, int N) {

    using BaseType = std::decay_t<std::remove_pointer_t<PtrType>>;
    static_assert(std::is_pointer_v<PtrType>, "Must be a pointer");
    static_assert(
        std::is_same_v<std::remove_cv_t<std::remove_pointer_t<PtrType>>, DType>,
        "Must point to DType");

    const auto projection_layout{get_tensor_layout<is_o>(batch_size, N)};
    const Tensor projection{
        make_tensor(make_gmem_ptr<DType>(g_ptr), projection_layout)};
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  COBRA_S_DEVICE auto identity_slice_head(int batch_size, int N) {
    const auto projection_layout{get_tensor_layout(batch_size, N)};
    const Tensor projection{make_identity_tensor(projection_layout.shape())};
    return projection(blockIdx.y, _, blockIdx.x, _);
  }

  // cp.async tile copy. CACHEGLOBAL bypasses L1 (16-byte only); CACHEALWAYS
  // is used for narrower granularities.
  template <typename LoadType> static constexpr auto get_tiled_copy() {

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

    if constexpr (sizeof(LoadType) == 16) {
      return make_tiled_copy(
          Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<LoadType>, DType>{},
          Layout<Shape<RowType, TPRType>, Stride<TPRType, _1>>{},
          Layout<Shape<_1, EPLType>>{});
    } else {
      return make_tiled_copy(
          Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<LoadType>, DType>{},
          Layout<Shape<RowType, TPRType>, Stride<TPRType, _1>>{},
          Layout<Shape<_1, EPLType>>{});
    }
  }

  static constexpr auto get_tiled_mma() {

    static_assert(thread_count % 32 == 0,
                  "thread_count must be a multiple of warp_size");

    using RowType = Int<thread_count / 32>;

    auto t_mma{
        make_tiled_mma(UniversalFMA<DType, DType, DType>{},
                       Layout<Shape<RowType, _32>,
                              Stride<_32, _1>>{})}; // 16x16x1 UniversalFMA

    return t_mma;
  }

  static_assert(B_c % B_r == 0, "B_c must be a multiple of B_r");

  // Predicated tile copy. With a cp.async copy atom this issues async copies
  // for in-bound rows and synchronously zero-fills out-of-bound rows. Both
  // become visible after the next cp_async_wait + __syncthreads.
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

  // Inner FMA loop only. Caller is responsible for making smem A and B
  // visible to all threads (cp_async_wait + __syncthreads).
  template <typename AEngineTypeMMA, typename ALayoutTypeMMA,
            typename BEngineTypeMMA, typename BLayoutTypeMMA,
            typename CEngineTypeMMA, typename CLayoutTypeMMA>
  COBRA_S_DEVICE void
  mma_fma_inner(const Tensor<AEngineTypeMMA, ALayoutTypeMMA> &a_mma_slice,
                const Tensor<BEngineTypeMMA, BLayoutTypeMMA> &b_mma_slice,
                Tensor<CEngineTypeMMA, CLayoutTypeMMA> &c_frag) {

    constexpr size_t mma_m_len{size(get<1>(ALayoutTypeMMA{}))};
    constexpr size_t mma_n_len{size(get<1>(BLayoutTypeMMA{}))};
    constexpr size_t mma_k_len{size(get<2>(BLayoutTypeMMA{}))};

    constexpr size_t elements_per_load{sizeof(VectorizedLoadType) /
                                       sizeof(TensorDType)};

    constexpr size_t slice_factor{mma_m_len};
    constexpr size_t mma_m_size{mma_m_len / slice_factor};

    float4 a_vecs[mma_m_size];
    float4 b_vecs[mma_n_len];

#pragma unroll 8
    for (size_t k{0}; k < mma_k_len; k += elements_per_load) {

      CUTE_UNROLL
      for (size_t m{0}; m < mma_m_len; m += mma_m_size) {

        CUTE_UNROLL
        for (size_t m_local{0}; m_local < mma_m_size; m_local++) {
          a_vecs[m_local] =
              *reinterpret_cast<float4 *>(&a_mma_slice(0, m + m_local, k));
        }

        CUTE_UNROLL
        for (size_t n{0}; n < mma_n_len; n++) {
          b_vecs[n] = *reinterpret_cast<float4 *>(&b_mma_slice(0, n, k));
        }

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
      const DType scale, const int bound, const int start_pos = 0) {

    static_assert(rank_v<ScoresTensorLayoutType> == 3,
                  "Per Register Attention scores must be 3 dimensional (mma, "
                  "mma_m, mma_n)");

    static_assert(rank_v<MaxTensorLayoutType> == 1,
                  "Per register, row maxes, muse be 1 dimensional");

    using MMAShape = decltype(get<0>(ScoresTensorLayoutType{}));
    constexpr size_t mma_m{size(get<1>(ScoresTensorLayoutType{}))};

    static_assert(rank(MMAShape{}) == 1, "not yet implemented");

    CUTE_UNROLL
    for (size_t m{0}; m < mma_m; ++m) {

      auto r_score_slice{r_scores(_, m, _)};
      auto scores_idty_slice{scores_idty_tensor(_, m, _)};
      auto p_slice{(prob_tensor(_, m, _))};
      auto o_slice{(out_tensor(_, m, _))};

      auto &current_max{max_tensor(m)};
      auto old_max{current_max};
      auto &current_sum{sum_tensor(m)};

      constexpr size_t slice_size{size(r_score_slice)};

      int adjusted_bound;

      if constexpr (causal_mask) {
        adjusted_bound = get<0>(scores_idty_slice(0)) + start_pos + 1;
      } else if constexpr (predicate) {
        adjusted_bound = bound;
      } else {
        adjusted_bound = 0;
      }

      CUTE_UNROLL
      for (size_t idx{0}; idx < slice_size; ++idx) {
        if constexpr (predicate || causal_mask) {
          auto n{get<1>(scores_idty_slice(idx))};
          if (n < adjusted_bound) {
            r_score_slice(idx) = r_score_slice(idx) * scale;
          } else {
            r_score_slice(idx) = -INFINITY;
          }
        } else {
          r_score_slice(idx) = r_score_slice(idx) * scale;
        }

        current_max = cuda::std::max(r_score_slice(idx), current_max);
      }

      current_max = warp_max(current_max);

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

      current_sum = current_sum * scale_old;

      DType local_sum{0};

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
        p_slice(idx) = p_score;
        r_score_slice(idx) = 0;
      }

      current_sum += warp_sum(local_sum);

      CUTE_UNROLL
      for (size_t i{0}; i < size(o_slice); i++) {
        o_slice(i) *= scale_old;
      }
    }
  }

  // start_pos is the KV-cache offset: it tells the attention kernel that
  // the current query tokens sit at position [start_pos, start_pos + N_q)
  // in the full sequence, and the KV cache already holds tokens [0, start_pos).
  void operator()(DType *Q, DType *K, DType *V, DType *O, uint32_t batch_size,
                  uint32_t N_q, uint32_t N_kv, uint32_t start_pos) {
    dim3 grid_dim{head_count, batch_size, ceil_div(N_q, B_r)};

    dim3 block_dim{thread_count};

    DType scale{rsqrt(static_cast<DType>(head_dim))};

    const auto tc_qk{get_tiled_copy<VectorizedLoadType>()};
    const auto tc_v{get_tiled_copy<ScalarLoadType>()};

    const auto tmma{get_tiled_mma()};

    auto kernel_fptr{pipelined::mha_kernel<Self, decltype(tc_qk),
                                           decltype(tc_v), decltype(tmma)>};

    size_t smem_size{sizeof(SharedStorage)};

    cudaFuncSetAttribute(
        kernel_fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaFuncSetAttribute(kernel_fptr,
                         cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kernel_fptr<<<grid_dim, block_dim, smem_size>>>(
        Q, K, V, O, N_q, N_kv, start_pos, scale, tc_qk, tc_v, tmma);
  }
};

} // namespace cobraml::kernels
