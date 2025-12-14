#pragma once
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

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
    template<typename MHAType>
    __global__ void mha_kernel(
        const typename MHAType::TensorDType * __restrict__ Q, 
        const typename MHAType::TensorDType * __restrict__ K, 
        const typename MHAType::TensorDType * __restrict__ V,
        typename MHAType::TensorDType * __restrict__ O,
        const int N
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

        // https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

        constexpr typename MHAType::HeadDimType d{};
        constexpr typename MHAType::QueryRowsType B_r{};
        constexpr typename MHAType::KVColsType B_c{};

        auto coord{make_coord(_, 0)};
        
        auto q_tiler{make_shape(B_r, d)};
        auto kv_tiler{make_shape(B_c, d)};

        Tensor q_iterator{local_tile(q_head, q_tiler, coord)};  // (B_r, d, floor(N / B_r))
        Tensor k_iterator{local_tile(k_head, kv_tiler, coord)}; // (B_r, d, floor(N / B_c))
        Tensor v_iterator{local_tile(v_head, kv_tiler, coord)}; // (B_r, d, floor(N / B_c))

        auto q_iters{size<2>(q_iterator)};
        auto kv_iters{size<2>(k_iterator)};

        if (thread0()){
            print(q_tensor);
            print("\n");
            print(q_iterator);
            print("\n");
            print(q_iters);
            print("\n");
            print(kv_iters);
        }

        for (size_t i{0}; i < q_iters; ++i){

            Tensor q_slice{q_iterator(_, _, i)};
            for (size_t j{0}; j < kv_iters; ++j){
                Tensor k_slice{k_iterator(_, _, i)};
                Tensor v_slice{v_iterator(_, _, i)};
            }
        }
        
    }
}

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

    static_assert(B_c % B_r == 0, "B_c must be a multiple of B_r");

    void operator()(
        DType * Q,
        DType * K,
        DType * V,
        DType * O,
        int batch_size,
        int N
    ){
        dim3 grid_dim{
            head_count,
            batch_size,
            ceil_div(N, B_r)
        };

        dim3 block_dim{
            thread_count
        };

        auto tc_q{get_tiled_copy()};
        auto tc_k{get_tiled_copy()};
        auto tc_v{get_tiled_copy()};

        naive::mha_kernel<Self><<<grid_dim, block_dim>>>(
            Q, K, V, O, N
        );

        // print(tc_q);

        print_latex(tc_q);
    }
    
};


}