//
// Created by root on 8/25/25.
//

#ifndef COBRAML_GEMM_CUH
#define COBRAML_GEMM_CUH
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

template <class TypeA, class LayoutA,
          class TypeB, class LayoutB,
          class TypeC, class LayoutC,
          class Alpha, class Beta>
void gemm_host_f16xf16_f32_tnt_TMA_UMMA(
    TypeA const * device_ptr_A,
    LayoutA layout_A,
    TypeB const * device_ptr_B,
    LayoutB const * layout_B,
    TypeC const * device_ptr_C,
    LayoutC const * layout_C,
    Alpha const alpha,
    Beta const beta
) {
    using namespace cute;

    assert(shape<0>(layout_A) == shape<0>(layout_C)); // share the same M
    assert(shape<0>(layout_B) == shape<1>(layout_C)); // share the same N
    assert(shape<1>(layout_B) == shape<1>(layout_A)); // share the same K

    Tensor global_A{make_tensor(make_gmem_ptr(device_ptr_A), layout_A)};
    Tensor global_B{make_tensor(make_gmem_ptr(device_ptr_B), layout_B)};
    Tensor global_C{make_tensor(make_gmem_ptr(device_ptr_C), layout_C)};

    auto M{shape<0>(layout_C)};
    auto N{shape<1>(layout_C)};
    auto K{shape<1>(layout_A)};

    TiledMMA tiled_mma{
        make_tiled_mma(
            SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K, UMMA::Major::K>{}
        )
    };


}

#endif //COBRAML_GEMM_CUH