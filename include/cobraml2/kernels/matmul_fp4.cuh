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

template<typename AType, typename BType, typename ConfigType>
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
    typename CopyAtomAType,
    typename CopyAtomBType,
    typename TensorAType,
    typename TensorBType,
    typename TensorCType,
    typename ConfigType
>
__global__ void gemm_kernel(
    __grid_constant__ CopyAtomAType const copy_a,
    __grid_constant__ CopyAtomBType const copy_b,
    TensorAType a,
    TensorBType b,
    TensorCType c,
    size_t M,
    ConfigType config
){
    using AType = typename TensorAType::value_type;
    using BType = typename TensorBType::value_type;
    using CType = typename TensorCType::value_type;

    using SharedStorageType = SharedStorage<AType, BType, ConfigType>;

    using ProducerViewType = typename ConfigType::ProducerViewType;
    using ConsumerViewType = typename ConfigType::ConsumerViewType;

    using ThreadRoleType = typename ThreadRoleManager::ThreadRole;

    using PipelineType = cobraml::pipelines::TwoWayPipeline<ProducerViewType, ConsumerViewType, ThreadRoleType>;

    extern __shared__ char shared_memory[];
    ThreadRoleType thread_role{ThreadRoleManager::assign_role(threadIdx.x)};

    using LoadTypeA = typename ProducerViewType::LoadTypeA;
    using LoadTypeB = typename ProducerViewType::LoadTypeB;

    SharedStorageType& shared_storage{*reinterpret_cast<SharedStorageType *>(shared_memory)};
    constexpr auto tiled_mma{config.tiled_mma};

    // one thread from the producer prefetches both tmas if it is tma
    if (thread_role == ThreadRoleType::Producer){
        if (elect_one_sync()){
            LoadTypeA::post_init(copy_a);
            LoadTypeB::post_init(copy_b);
        }
    }

    constexpr Layout cluster_vmnk_layout{
        config.cluster_layout_vmnk
    };

    auto coord_vmnk{make_coord(
        blockIdx.x % size<0>(cluster_vmnk_layout), // Peer CTA coordinate, when 2sm this flips between 0 and 1 
        blockIdx.x / size<0>(cluster_vmnk_layout), //    MMA-M coordinate
        blockIdx.y,                                //    MMA-N coordinate
        _                                          //    MMA-K coordinate
    )};
    
    auto coord_slice{
        select<1, 2, 3>(coord_vmnk)
    };

    auto mma_v = get<0>(coord_vmnk);
    ThrMMA cta_mma{tiled_mma.get_slice(mma_v)};   // Use Peer CTA coordinate

    // BM, BN, BK
    constexpr auto mma_tiler{
        config.mma_tiler
    };

    Tensor global_a_iter{
        local_tile(
            a, mma_tiler, coord_slice, Step<_1, X, _1>{} 
        )
    };

    Tensor global_b_iter{
        local_tile(
            b, mma_tiler, coord_slice, Step<X, _1, _1>{} 
        )
    };

    Tensor global_c_iter{
        local_tile(
            b, mma_tiler, coord_slice, Step<_1, _1, X>{} 
        )
    };

    if (thread0()){
        print(cta_mma.partition_A(global_a_iter)); print("\n"); print(global_b_iter); print("\n"); print(global_c_iter);
    }

    // slice out the local tile from gmem
    // partition using the tiled_mma partition_A(gA)
    // this can then be passed into tma_partition
    

    // cutlass example 

    /**
     *   1. load_init() (setup, once) — lines 527-543:
  // tma_partition happens here, returns LoadParams struct
  auto [tAgA_mkl, tAsA] = tma_partition(*tma_load_a_, ...);
  auto [tBgB_nkl, tBsB] = tma_partition(*tma_load_b_, ...);
  return LoadParams{tAgA_mkl, tBgB_nkl, tAsA, tBsB, mcast_masks...};

  2. load() (producer mainloop) — lines 586-626:
  // unpack the partitioned tensors
  auto [tAgA_mkl, tBgB_nkl, tAsA, tBsB, ...] = load_inputs;

  // slice to this CTA's M/N coord
  Tensor tAgA = tAgA_mkl(_, m_coord, _, l_coord);
  Tensor tBgB = tBgB_nkl(_, n_coord, _, l_coord);

  auto barrier_token = pipeline.producer_try_acquire(state);  // <-- your check_barrier

  while (k_tile_count > 0) {
      pipeline.producer_acquire(state, barrier_token);         // <-- your get_tiles wait

      BarrierType* tma_barrier = pipeline.producer_get_barrier(state);  // get full_barrier[i]
      int write_stage = state.index();
      ++state;
      barrier_token = pipeline.producer_try_acquire(state);    // prefetch next barrier

      if (elect_one_sync()) {                                  // only leader
          copy(tma_a->with(*tma_barrier, mcast_mask_a), tAgA(_, k_tile), tAsA(_, write_stage));
          copy(tma_b->with(*tma_barrier, mcast_mask_b), tBgB(_, k_tile), tBsB(_, write_stage));
      }
      --k_tile_count;
      ++k_tile_iter;
  }
     * 
     */

    PipelineType pipeline(
        shared_storage.full_barrier, 
        shared_storage.empty_barrier,
        ThreadRoleType::Producer,
        ThreadRoleType::Consumer,  
        thread_role
    );
}

struct GEMM {

    template<
        typename AType,
        typename BType,
        typename CType,
        typename ConfigType,
        size_t static_N,
        size_t static_K
    >
    void operator()(
        const AType * __restrict__ a,
        const BType *__restrict__ b,
        CType *__restrict__ c,
        size_t M,
        const GEMMShapeManager<AType, BType, CType, static_N, static_K> &shape_manager,
        const ConfigType &config
    ) {

        using ProducerViewType = typename ConfigType::ProducerViewType;
        using LoadTypeA = typename ProducerViewType::LoadTypeA;
        using LoadTypeB = typename ProducerViewType::LoadTypeB;

        auto [tensor_a, tensor_b, tensor_c] = shape_manager.init_tensors(a, b, c);

        auto copy_atom_a{LoadTypeA::create_copy_atom(
            tensor_a,
            config.a_smem_layout(_,_,_,cute::Int<0>{}),
            config.mma_tiler,
            config.tiled_mma,
            config.cluster_layout_vmnk
        )};

        auto copy_atom_b{LoadTypeB::create_copy_atom(
            tensor_b,
            config.b_smem_layout(_,_,_,cute::Int<0>{}),
            config.mma_tiler,
            config.tiled_mma,
            config.cluster_layout_vmnk
        )};

        constexpr dim3 block_dim{ThreadRoleManager::total_warps * 32, 1, 1};
        const dim3 grid_dim{
            static_cast<uint32_t>(ceil_div(M, config.bM) * get<0>(config.cluster_shape)),
            static_cast<uint32_t>(ceil_div(static_N, config.bN) * get<1>(config.cluster_shape))
        };

        const dim3 cluster_dim{
            get<0>(config.cluster_shape),
            get<1>(config.cluster_shape),
            get<2>(config.cluster_shape)
        };

        using SharedStorageType = SharedStorage<AType, BType, ConfigType>;
        constexpr size_t smem_size{sizeof(SharedStorageType)};

        void const* kernel = (void const*) gemm_kernel<
            decltype(copy_atom_a),
            decltype(copy_atom_b),
            decltype(tensor_a),
            decltype(tensor_b),
            decltype(tensor_c),
            ConfigType>;

        void* kernel_params[] = {
            &copy_atom_a,
            &copy_atom_b,
            &tensor_a,
            &tensor_b,
            &tensor_c,
            &M,
            const_cast<ConfigType*>(&config)
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
