#pragma once
#include "../algos.cuh"
#include "../utilities.cuh"
#include "../macros.cuh"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

namespace cobraml::kernels {
using namespace cute;


namespace SM100 {

template<
    typename TmaDescriptorAType, 
    typename TmaDescriptorBType, 
    typename TensorAType, 
    typename TensorBType, 
    typename TensorCType,
    typename ConfigType
>
__global__ void gemm_device(
    __grid_constant__ TmaDescriptorAType const tma_a,
    __grid_constant__ TmaDescriptorBType const tma_b,
    TensorAType a,
    TensorBType b,
    TensorCType c,
    size_t M,
    ConfigType config
){
    if (thread0()){
        print(a); print("\n");
        print(b); print("\n");
        print(c); print("\n");
    }
}

template<
    typename AType, 
    typename BType, 
    typename CType
>
struct GEMM {

    static_assert(std::is_same_v<AType, BType>, "AType and BType should be the same");

    template<
        size_t N, 
        size_t K,
        size_t MMA_M,
        size_t MMA_N,
        size_t pipeline_stages,
        size_t MMA_K_per_tile = 4
    >
    struct Config {

        using SwizzleType = UMMA::Layout_K_SW128_Atom<AType>;
        constexpr static size_t static_N{N};
        constexpr static size_t static_K{K};
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

        using CopyAtomA = SM90_TMA_LOAD;
        using CopyAtomB = SM90_TMA_LOAD;

        static constexpr auto cluster_shape{make_shape(Int<1>{}, Int<1>{}, Int<1>{})};
        static constexpr Layout cluster_layout_vmnk{tiled_divide(make_layout(cluster_shape), make_tile(typename decltype(tiled_mma)::AtomThrID{}))};
    };

    template<typename ConfigType>
    struct SharedStorage {
        // Define shared memory buffers for A and B tiles
        alignas(128) cute::ArrayEngine<AType, size(ConfigType::a_smem_layout)> smem_a;
        alignas(128) cute::ArrayEngine<BType, size(ConfigType::b_smem_layout)> smem_b;

        alignas(16) cute::ArrayEngine<cute::uint64_t, ConfigType::static_pipeline_stages> full_barrier;
        alignas(16) cute::ArrayEngine<cute::uint64_t, ConfigType::static_pipeline_stages> empty_barrier;
    };

    template<typename ConfigType>
    auto init_tensors(
        const AType * __restrict__ a,
        const BType *__restrict__ b,
        CType *__restrict__ c,
        size_t M,
        ConfigType config
    ){

        using NType = Int<ConfigType::static_N>;
        using KType = Int<ConfigType::static_K>;

        auto layout_a{
            make_layout(make_shape(M, KType{}), LayoutRight{})
        };
        auto layout_b{
            make_layout(make_shape(NType{}, KType{}), LayoutRight{})
        };
        auto layout_c{
            make_layout(make_shape(M, NType{}), LayoutRight{})
        };

        auto a_tensor{
            make_tensor(make_gmem_ptr<AType>(a), layout_a)
        };
        auto b_tensor{
            make_tensor(make_gmem_ptr<BType>(b), layout_b)
        };
        auto c_tensor{
            make_tensor(make_gmem_ptr<CType>(c), layout_c)
        };

        return cute::make_tuple(a_tensor, b_tensor, c_tensor);
    }

    template<typename ConfigType>
    void operator()(
        const AType * __restrict__ a,
        const BType *__restrict__ b,
        CType *__restrict__ c,
        size_t M,
        ConfigType config) {

        auto [tensor_a, tensor_b, tensor_c] = init_tensors(
            a, b, c, M, config
        );

        auto tma_descriptor_a{make_tma_atom_A_sm100<AType>(
            typename ConfigType::CopyAtomA{},
            tensor_a,
            ConfigType::a_smem_layout(_,_,_,cute::Int<0>{}),
            ConfigType::mma_tiler,
            ConfigType::tiled_mma,
            ConfigType::cluster_layout_vmnk)};

        auto tma_descriptor_b{make_tma_atom_B_sm100<BType>(
            typename ConfigType::CopyAtomB{},
            tensor_b,
            ConfigType::b_smem_layout(_,_,_,cute::Int<0>{}),
            ConfigType::mma_tiler,
            ConfigType::tiled_mma,
            ConfigType::cluster_layout_vmnk)};

        constexpr size_t scheduler_warp{0};
        constexpr size_t epilogue_warp{0};
        constexpr size_t load_warp{1};
        constexpr size_t mma_warp{1};

        constexpr size_t total_warps{
            scheduler_warp + epilogue_warp + load_warp + mma_warp
        };

        constexpr dim3 block_dim{total_warps * 32, 1, 1};
        const dim3 grid_dim{
            static_cast<uint32_t>(ceil_div(M, ConfigType::bM) * get<0>(ConfigType::cluster_shape)),
            static_cast<uint32_t>(ceil_div(ConfigType::static_N, ConfigType::bN) * get<1>(ConfigType::cluster_shape))
        };

        const dim3 cluster_dim{
            get<0>(ConfigType::cluster_shape),
            get<1>(ConfigType::cluster_shape),
            get<2>(ConfigType::cluster_shape)
        };

        using SharedStorageType = SharedStorage<ConfigType>;
        constexpr size_t smem_size{sizeof(SharedStorageType)};

        void const* kernel = (void const*) gemm_device<
            decltype(tma_descriptor_a),
            decltype(tma_descriptor_b),
            decltype(tensor_a),
            decltype(tensor_b),
            decltype(tensor_c),
            ConfigType>;

        void* kernel_params[] = {
            &tma_descriptor_a,
            &tma_descriptor_b,
            &tensor_a,
            &tensor_b,
            &tensor_c,
            &M,
            &config
        };

        cudaStream_t stream = nullptr;

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        assert(grid_dim.x % cluster_dim.x == 0 &&
               grid_dim.y % cluster_dim.y == 0 &&
               grid_dim.z % cluster_dim.z == 0 &&
               "Grid dimensions must be divisible by cluster dimensions");

        cudaLaunchConfig_t launch_config{};
        launch_config.gridDim = grid_dim;
        launch_config.blockDim = block_dim;
        launch_config.dynamicSmemBytes = smem_size;
        launch_config.stream = stream;

        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {cluster_dim.x, cluster_dim.y, cluster_dim.z};
        attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[1].val.programmaticStreamSerializationAllowed = 0;

        launch_config.attrs = attrs;
        launch_config.numAttrs = 2;

        auto status = cudaLaunchKernelExC(&launch_config, kernel, kernel_params);
        assert(status == cudaSuccess && "Kernel launch failed");
    }
};

}
}
