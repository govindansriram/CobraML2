#pragma once
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include "./pipelines.cuh"

namespace cobraml::configs {
using namespace cute;

namespace sm100{

template<
    typename AType,
    typename BType,
    typename CType,
    typename LoadPipelineType,
    typename MMAPipelineType,
    size_t MMA_M,
    size_t MMA_N,
    size_t pipeline_stages
>
struct GemmConfigTmaUmma {

    static constexpr size_t MMA_K_per_tile = 4;

    using SwizzleType = UMMA::Layout_K_SW128_Atom<AType>;
    constexpr static size_t static_pipeline_stages{pipeline_stages};

    constexpr static size_t swizzle_bits{128};

    constexpr static auto get_tiled_mma(){
        static_assert(std::is_same_v<AType, cutlass::bfloat16_t>, "AType should be cutlass::bfloat16_t");
        static_assert(std::is_same_v<BType, cutlass::bfloat16_t>, "BType should be cutlass::bfloat16_t");
        static_assert(std::is_same_v<CType, cutlass::bfloat16_t> || std::is_same_v<CType, float>, "CType should be cutlass::bfloat16_t or float");

        return make_tiled_mma(
            SM100_MMA_F16BF16_SS<
                AType, BType, CType,                 
                MMA_M, MMA_N,                            
                UMMA::Major::K, UMMA::Major::K>{}
        );  
    }

    constexpr static auto tiled_mma{get_tiled_mma()};

    template<bool is_a=true, typename XType, typename YType>
    constexpr static auto get_smem_layout(XType x, YType y) {
        auto mma_shape{
            [&]{
                if constexpr (is_a) {
                    return partition_shape_A(tiled_mma, make_shape(x, y * Int<MMA_K_per_tile>{}));
                }else {
                    return partition_shape_B(tiled_mma, make_shape(x, y * Int<MMA_K_per_tile>{}));
                }
            }()
        };

        auto pipelined_mma_shape{
            append(mma_shape, Int<pipeline_stages>{})
        };

        return UMMA::tile_to_mma_shape(SwizzleType{}, pipelined_mma_shape);
    }

    constexpr static auto bM{tile_size<0>(tiled_mma)};                            
    constexpr static auto bN{tile_size<1>(tiled_mma)};
    constexpr static auto bK{tile_size<2>(tiled_mma)};
    constexpr static auto mma_tiler{make_shape(bM, bN, bK * Int<MMA_K_per_tile>{})};                             

    static_assert((bK * sizeof_bits<AType>::value * MMA_K_per_tile) % swizzle_bits == 0, "bK must be a multiple of swizzle_bits");
                                                                                    // For 16b types, tcgen05.mma has K16.

    constexpr static auto a_smem_layout{get_smem_layout<true>(bM, bK)};
    constexpr static auto b_smem_layout{get_smem_layout<false>(bN, bK)};

    static constexpr auto cluster_shape{make_shape(Int<1>{}, Int<1>{}, Int<1>{})};
    static constexpr Layout cluster_layout_vmnk{tiled_divide(make_layout(cluster_shape), make_tile(typename decltype(tiled_mma)::AtomThrID{}))};
};
}

}