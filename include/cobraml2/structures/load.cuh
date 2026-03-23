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
        const SmemLayoutType a_smem_layout,
        const MMATilerShapeType mma_tiler,
        const TiledMMAType tiled_mma,
        const ClusterLayoutType cluster_layout
    ){
        if constexpr(is_A){
            return make_tma_atom_A_sm100<DType>(
                CopyAtom{},
                gmem_tensor,
                a_smem_layout,
                mma_tiler,
                tiled_mma,
                cluster_layout
            );
        }else{
            return make_tma_atom_B_sm100<DType>(
                CopyAtom{},
                gmem_tensor,
                a_smem_layout,
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
};

}

}