#pragma once
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include "../macros.cuh"

namespace cobraml::loadop{

namespace sm100 {
using namespace cute;

template<typename DType, bool is_A>
struct TMALoad{

    using CopyAtom = SM90_TMA_LOAD;

    template<
        typename TensorType,
        typename SmemLayoutType,
        typename MMATilerShapeType,
        typename TiledMMAType,
        typename ClusterLayoutType
    >
    static auto create_copy_atom(
        TensorType &gmem_tensor,
        const SmemLayoutType smem_layout,
        const MMATilerShapeType mma_tiler,
        const TiledMMAType tiled_mma,
        const ClusterLayoutType cluster_layout
    ){
        if constexpr(is_A){
            return make_tma_atom_A_sm100<DType>(
                CopyAtom{},
                gmem_tensor,
                smem_layout,
                mma_tiler,
                tiled_mma,
                cluster_layout
            );
        }else{
            return make_tma_atom_B_sm100<DType>(
                CopyAtom{},
                gmem_tensor,
                smem_layout,
                mma_tiler,
                tiled_mma,
                cluster_layout
            );
        }
    }

    template<typename TmaAtomType>
    COBRA_S_DEVICE void post_init(TmaAtomType &atom){
        cute::prefetch_tma_descriptor(atom.get_tma_descriptor());
    }


    /**
     * @brief Partition global and shared tensors for TMA copy. Groups all modes
     * except the last (K tiles / pipeline stages) into a single mode since
     * tma_partition expects 2 dimensions. No multicast support yet.
     *
     * @param tma_atom the TMA atom created by create_copy_atom
     * @param global_tensor partitioned global memory tensor
     * @param shared_tensor partitioned shared memory tensor
     * @return tuple of (global_partition, shared_partition)
     */
    template<
        typename TmaAtomType,
        typename GlobalTensorLayoutType,
        typename GlobalTensorEngineType,
        typename SharedTensorLayoutType,
        typename SharedTensorEngineType
    >
    COBRA_S_DEVICE static auto partition_copy_pair(
        TmaAtomType const &tma_atom,
        Tensor<GlobalTensorEngineType, GlobalTensorLayoutType> global_tensor,
        Tensor<SharedTensorEngineType, SharedTensorLayoutType> shared_tensor
    ){
        constexpr size_t group_shared_rank{rank_v<SharedTensorLayoutType> - 1};
        constexpr size_t group_global_rank{rank_v<GlobalTensorLayoutType> - 1};

        // TODO add multicast support
        auto [global_part, shared_part]{tma_partition(tma_atom,
            Int<0>{}, Layout<_1>{},
            group_modes<0,group_shared_rank>(shared_tensor), group_modes<0,group_global_rank>(global_tensor))
        };

        return cute::make_tuple(global_part, shared_part);
    }
};

}

}