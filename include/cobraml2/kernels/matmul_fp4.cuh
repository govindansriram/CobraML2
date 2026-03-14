#pragma once
#include "../algos.cuh"
#include "../utilities.cuh"
#include "../macros.cuh"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

namespace cobraml::SM100::kernels {
using namespace cute;

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
);

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
        using ProducerBarrierType = cutlass::arch::ClusterTransactionBarrier;
        using ConsumerBarrierType = cutlass::arch::ClusterBarrier;

        // Define shared memory buffers for A and B tiles
        alignas(128) cute::ArrayEngine<AType, size(ConfigType::a_smem_layout)> smem_a;
        alignas(128) cute::ArrayEngine<BType, size(ConfigType::b_smem_layout)> smem_b;

        alignas(16) ProducerBarrierType full_barrier[ConfigType::static_pipeline_stages];
        alignas(16) ConsumerBarrierType empty_barrier[ConfigType::static_pipeline_stages];
    };

    enum class ThreadRole {
        Producer,
        Consumer
    };

    COBRA_S_DEVICE ThreadRole get_thread_role() {
        int warp_idx = threadIdx.x / 32;
        return (warp_idx == 0) ? ThreadRole::Producer : ThreadRole::Consumer;
    }

    template<typename ConfigType>
    struct Pipeline {

        struct State {
            int index{0};
            uint32_t phase{0};

            COBRA_DEVICE State& operator++() {
                ++index;
                if (index >= ConfigType::static_pipeline_stages) {
                    index = 0;
                    phase ^= 1;
                }

                return *this;
            }
        };

        enum class BarrierStatus : uint32_t {
            WaitDone = 1,
            WaitAgain = 0
        };

        SharedStorage<ConfigType>& shared_storage;
        ThreadRole role;

        struct ProducerView {

            SharedStorage<ConfigType>& shared_storage;
            bool is_leader{false};

            private:
            COBRA_DEVICE typename SharedStorage<ConfigType>::ConsumerBarrierType& get_barrier_status(State& state) {
                return shared_storage.full_barrier[state.index];
            }

            static constexpr size_t transaction_bytes{
                size(ConfigType::a_smem_layout(_, _, _, 0)) * sizeof(AType) + size(ConfigType::b_smem_layout(_, _, _, 0)) * sizeof(BType)
            };

            template<bool is_a>
            COBRA_DEVICE auto get_smem_tensor(){
                if constexpr (is_a) {   
                    return make_tensor(
                        make_smem_ptr(shared_storage.smem_a.begin()), 
                        ConfigType::a_smem_layout
                    );
                }else{
                    return make_tensor(
                        make_smem_ptr(shared_storage.smem_b.begin()), 
                        ConfigType::b_smem_layout
                    );
                }
            }

            public:
            COBRA_DEVICE ProducerView(SharedStorage<ConfigType>& shared_storage): shared_storage(shared_storage) {
                is_leader = (cute::elect_one_sync() == 1);
            }

            COBRA_DEVICE BarrierStatus check_barrier(State& state){
                return static_cast<BarrierStatus>(get_barrier_status(state).try_wait(state.phase));
            }

            COBRA_DEVICE auto wait_barrier(State& state, BarrierStatus status){
                if (status == BarrierStatus::WaitAgain) 
                    get_barrier_status(state).wait(state.phase);

                if (is_leader)
                    get_barrier_status(state).arrive_and_expect_tx(transaction_bytes);

                return make_tuple(
                    get_smem_tensor<true>()(_, _, _, state.index), 
                    get_smem_tensor<false>()(_, _, _, state.index)
                );
            }
        };

        COBRA_DEVICE Pipeline(
            SharedStorage<ConfigType>& shared_storage,
            const ThreadRole& role) : shared_storage(shared_storage), role(role) {
            if (role == ThreadRole::Producer) {
                if (elect_one_sync()) {
                    for (size_t i = 0; i < ConfigType::static_pipeline_stages; ++i) {
                        shared_storage.full_barrier[i].init(1);
                        shared_storage.empty_barrier[i].init(1);
                    }
                }
            }
            cutlass::arch::fence_barrier_init();
            __syncthreads();
        }

        COBRA_DEVICE State init_producer_state() {
            return State{0, 1};
        }

        COBRA_DEVICE State init_consumer_state() {
            return State{0, 0};
        }

        COBRA_DEVICE ProducerView producer_view() {
            return ProducerView(shared_storage);
        }

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
    using AType = typename TensorAType::value_type;
    using BType = typename TensorBType::value_type;
    using CType = typename TensorCType::value_type;
    using GemmType = GEMM<AType, BType, CType>;
    using SharedStorageType = typename GemmType::template SharedStorage<ConfigType>;
    using PipelineType = typename GemmType::template Pipeline<ConfigType>;
    using ThreadRoleEnum = typename GemmType::ThreadRole;

    extern __shared__ char shared_memory[];

    SharedStorageType& shared_storage{*reinterpret_cast<SharedStorageType *>(shared_memory)};
    ThreadRoleEnum role{GemmType::get_thread_role()};
    PipelineType pipeline(shared_storage, role);

    if (role == ThreadRoleEnum::Producer) {
        auto starting_state{pipeline.init_producer_state()};
        auto producer_view{pipeline.producer_view()};
    }
}


}
