#pragma once
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cub/cub.cuh>
#include "../algos.cuh"


namespace cobraml::kernels{

using namespace cute;

namespace naive{

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
    template<
        typename MHAType,
        typename TiledCopyQType,
        typename TiledCopyKType,
        typename TiledCopyVType,
        typename TiledMMA
    >
    __global__ void mha_kernel(
        const typename MHAType::TensorDType * __restrict__ Q, 
        const typename MHAType::TensorDType * __restrict__ K, 
        const typename MHAType::TensorDType * __restrict__ V,
        typename MHAType::TensorDType * __restrict__ O,
        const int N,
        TiledCopyQType tc_q,
        TiledCopyKType tc_k,
        TiledCopyVType tc_v,
        TiledMMA t_mma_qk
    ){

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
        SharedStorageType * shared_storage{reinterpret_cast<SharedStorageType *>(shared_memory)};

        auto shared_q_ptr{make_smem_ptr(shared_storage->Q.begin())};
        auto shared_k_ptr{make_smem_ptr(shared_storage->K.begin())};
        auto shared_v_ptr{make_smem_ptr(shared_storage->V.begin())};

        Tensor shared_q{
            make_tensor(shared_q_ptr, typename SharedStorageType::QLayoutType{})
        };

        Tensor shared_k{
            make_tensor(shared_k_ptr, typename SharedStorageType::KVLayoutType{})
        };

        Tensor shared_v{
            make_tensor(shared_k_ptr, typename SharedStorageType::KVLayoutType{})
        };

        // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

        constexpr typename MHAType::HeadDimType d{};
        constexpr typename MHAType::QueryRowsType B_r{};
        constexpr typename MHAType::KVColsType B_c{};

        auto coord{make_coord(_, 0)};
        
        auto q_tiler{make_shape(B_r, d)};
        auto kv_tiler{make_shape(B_c, d)};

        Tensor q_iterator{local_tile(q_head, q_tiler, coord)};  // (B_r, d, floor(N / B_r))
        Tensor k_iterator{local_tile(k_head, kv_tiler, coord)}; // (B_c, d, floor(N / B_c))
        Tensor v_iterator{local_tile(v_head, kv_tiler, coord)}; // (B_c, d, floor(N / B_c))

        auto iters{size<2>(k_iterator)};

        Tensor q_slice{q_iterator(_, _, blockIdx.z)};

        // move q to shared memory

        // t prefix means unique to this thread
        ThrCopy thr_copy_q{tc_q.get_slice(threadIdx.x)};
        ThrCopy thr_copy_k{tc_k.get_slice(threadIdx.x)};

        Tensor tQ_global_part{thr_copy_q.partition_S(q_slice)};
        Tensor tQ_shared_part{thr_copy_q.partition_D(shared_q)};
        Tensor tK_shared_part{thr_copy_k.partition_D(shared_k)};

        ThrMMA thr_mma_q{
            t_mma_qk.get_slice(threadIdx.x)
        };

        Tensor q_mma{thr_mma_q.partition_A(shared_q)};
        Tensor k_mma{thr_mma_q.partition_B(shared_k)};

        auto mma_m{select<1>(q_mma.shape())};
        auto mma_n{select<1>(k_mma.shape())};

        constexpr auto s_mma_layout{flatten(make_layout(make_shape(_1{}, mma_m, mma_n), LayoutRight{}))};
        Tensor s_mma_r{make_tensor<DType>(s_mma_layout)};
        clear(s_mma_r);  // Zero the accumulator

        copy(tc_q, tQ_global_part, tQ_shared_part);
        __syncthreads(); // redundant

        // start with the lowest possible value
        Tensor m{make_tensor<DType>(mma_m)};
        fill(m, cuda::std::numeric_limits<DType>::lowest());

        for (size_t iter{0}; iter < iters; ++iter){
            Tensor k_slice{k_iterator(_, _, iter)};
            Tensor v_slice{v_iterator(_, _, iter)};
            Tensor tK_global_part{thr_copy_k.partition_S(k_slice)};

            copy(tc_q, tK_global_part, tK_shared_part);

            __syncthreads();

            gemm(t_mma_qk, q_mma, k_mma, s_mma_r);

            // if (thread0()){
            //     print(m);
            //     print("--------------------------------------------\n");
            // }

            MHAType::row_max(m, s_mma_r);
        }
        
    }
}

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
template<
    int head_count, 
    int head_dim, 
    int B_r,
    int B_c,
    typename DType,
    int thread_count = 128
>
struct MHA{

    using TensorDType = DType;
    using Self = MHA<head_count, head_dim, B_r, B_c, DType, thread_count>;

    using NumHeadsType = Int<head_count>;
    using HeadDimType = Int<head_dim>;
    using QueryRowsType = Int<B_r>;
    using KVColsType = Int<B_c>;

    using VectorizedLoadType = uint128_t;

    struct SharedStorage{
        ArrayEngine<DType, B_r * head_dim> Q;
        ArrayEngine<DType, B_c * head_dim> K;
        ArrayEngine<DType, B_c * head_dim> V;

        using QLayoutType = Layout<Shape<QueryRowsType, HeadDimType>, Stride<HeadDimType, _1>>;
        using KVLayoutType = Layout<Shape<KVColsType, HeadDimType>, Stride<HeadDimType, _1>>;
    };

    static constexpr int threads_per_block{thread_count};

    CUTE_HOST_DEVICE static auto get_tensor_layout(size_t batch_size, size_t N){
        return make_layout(
            make_shape(batch_size, NumHeadsType{}, N, HeadDimType{}), LayoutRight{}
        );
    }

    CUTE_HOST_DEVICE static constexpr int get_total_threads(){
        constexpr int elements_per_load{sizeof(VectorizedLoadType) / sizeof(DType)};
        constexpr int threads_per_row{head_dim / elements_per_load};

        static_assert(
            head_dim % elements_per_load == 0, 
            "head dimension in bytes must be a multiple of 128"
        );

        constexpr int total_threads{B_r * threads_per_row};

        static_assert(
            total_threads % 32 == 0, 
            "block size requires an odd number of threads"
        );

        return total_threads;
    }

    static constexpr auto get_tiled_copy(){

        // ensures no repeat across the head dimension

        constexpr int elements_per_load{sizeof(VectorizedLoadType) / sizeof(DType)};
        constexpr int threads_per_row{head_dim / elements_per_load};

        static_assert(
            head_dim % threads_per_row == 0, 
            "the head dimension is not a multiple of 128 bytes"
        );

        using TPRType = Int<threads_per_row>;
        using EPLType = Int<elements_per_load>;
        constexpr int rows{thread_count / threads_per_row};
        using RowType = Int<rows>;

        static_assert(
            thread_count % threads_per_row == 0, 
            "thread_count is not compatible with this head dimension"
        );

        static_assert(
            B_r % rows == 0, 
            "threads load too many rows B_r must be increased"
        );

        return make_tiled_copy(
            Copy_Atom<UniversalCopy<VectorizedLoadType>, DType>{},
            Layout<Shape<RowType, TPRType>,Stride<TPRType,_1>>{},
            Layout<Shape<_1, EPLType>>{}
        );
    }

    static constexpr auto get_tiled_mma(){

        static_assert(
            thread_count % 32 == 0,
            "thread_count must be a multiple of warp_size" 
        );

        using RowType = Int<thread_count / 32>;

        // one warp computes one row

        TiledMMA t_mma = make_tiled_mma(
            UniversalFMA<DType,DType,DType>{},
            Layout<Shape<RowType,_32>, Stride<_32, _1>>{}
        );  // 16x16x1 UniversalFMA
        
        return t_mma;
    }

    static_assert(B_c % B_r == 0, "B_c must be a multiple of B_r");

    template<
        typename MaxTensorEngineType,
        typename ScoresTensorEngineType, 
        typename MaxTensorLayoutType,
        typename ScoresTensorLayoutType
    >
    __device__ static void row_max(
        Tensor<MaxTensorEngineType, MaxTensorLayoutType> &max_tensor, 
        Tensor<ScoresTensorEngineType, ScoresTensorLayoutType> const &scores){

            static_assert(
                rank_v<ScoresTensorLayoutType> == 3,
                "Per Register Attention scores must be 3 dimensional (mma, mma_m, mma_n)"
            );

            static_assert(
                rank_v<MaxTensorLayoutType> == 1,
                "Per register, row maxes, muse be 1 dimensional"
            );

            constexpr size_t mma_m{size(get<1>(ScoresTensorLayoutType{}))};
            constexpr Layout mma_shape{get<0>(ScoresTensorLayoutType{})};
            
            CUTE_UNROLL
            for(size_t m{0}; m < mma_m; ++m){

                auto score_slice{scores(_, m, _)};

                if constexpr (rank(mma_shape) > 1){

                } else{
                    auto& current_max{max_tensor(m)};
                    constexpr size_t slice_size{size(score_slice)};

                    CUTE_UNROLL
                    for (size_t idx{0}; idx < slice_size; ++idx){
                        // uses hardware unit, removes warp divergence from branch checks
                        current_max = cuda::std::max(score_slice(idx), current_max);
                    }

                    current_max = warp_max(current_max);
                    // if (thread(31) && m == 0){
                    //     print(current_max);
                    // }
                }

            }
    }

    void operator()(
        DType * Q,
        DType * K,
        DType * V,
        DType * O,
        uint32_t batch_size,
        uint32_t N
    ){
        dim3 grid_dim{
            head_count,
            batch_size,
            ceil_div(N, B_r)
        };

        dim3 block_dim{
            thread_count
        };

        const auto tc_q{get_tiled_copy()};
        const auto tc_k{get_tiled_copy()};
        const auto tc_v{get_tiled_copy()};
        const auto t_mma{get_tiled_mma()};

        auto kernel_fptr{
            naive::mha_kernel<
                Self,
                decltype(tc_q),
                decltype(tc_k),
                decltype(tc_v),
                decltype(t_mma)
            >
        };

        size_t smem_size{sizeof(SharedStorage)};

        // Set L1 to be SMEM only
        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        kernel_fptr<<<grid_dim, block_dim, smem_size>>>(
            Q, K, V, O, N,
            tc_q, tc_k, tc_v,
            t_mma
        );

        // print(tc_q);
        // print_latex(tc_q);
    }
    
};


}