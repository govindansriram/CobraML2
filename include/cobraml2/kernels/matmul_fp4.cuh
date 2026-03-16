#pragma once
#include "../algos.cuh"
#include "../utilities.cuh"
#include "../macros.cuh"
#include "../structures/configs.cuh"
#include "../structures/pipelines.cuh"
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>

namespace cobraml::kernels::sm100 {
using namespace cute;

struct ThreadRoleManager{

    static constexpr uint warp_size{32};
    static constexpr uint total_warps{2};

    enum class ThreadRole {
        Producer = warp_size,
        Consumer = warp_size * 2
    };

    COBRA_S_DEVICE ThreadRole assign_role(uint thread_num){
        if (thread_num < static_cast<uint>(ThreadRole::Producer)){
            return ThreadRole::Producer;
        }else{
            return ThreadRole::Consumer;
        }
    }
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

template<
    typename TmaDescriptorAType,
    typename TmaDescriptorBType,
    typename TensorAType,
    typename TensorBType,
    typename TensorCType,
    typename ConfigType
>
__global__ void gemm_kernel(
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

    using ThreadRoleType = typename ThreadRoleManager::ThreadRole;

    extern __shared__ char shared_memory[];
    ThreadRoleType thread_role{ThreadRoleManager::assign_role(threadIdx.x)};

    SharedStorageType& shared_storage{*reinterpret_cast<SharedStorageType *>(shared_memory)};
    ThreadRoleEnum role{GemmType::get_thread_role()};
    PipelineType pipeline(shared_storage, role);

    if (role == ThreadRoleEnum::Producer) {
        auto starting_state{pipeline.init_producer_state()};
        auto producer_view{pipeline.producer_view()};
    }
}

struct GEMM {

    template<
        typename AType,
        typename BType,
        typename CType,
        typename CopyPipeline,
        typename MMAPipeline,
        size_t MMA_M,
        size_t MMA_N,
        size_t static_N,
        size_t static_K,
        size_t pipeline_stages
    >
    void operator()(
        const AType * __restrict__ a,
        const BType *__restrict__ b,
        CType *__restrict__ c,
        size_t M,
        const GEMMShapeManager<AType, BType, CType, static_N, static_K> &shape_manager,
        const configs::sm100::GemmConfigTmaUmma<AType, BType, CType, CopyPipeline, MMAPipeline, MMA_M, MMA_N, pipeline_stages> &config
    ) {

        auto [tensor_a, tensor_b, tensor_c] = shape_manager.init_tensors(a, b, c);

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
