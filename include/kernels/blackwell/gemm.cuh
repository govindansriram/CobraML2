//
// Created by root on 8/25/25.
//

#ifndef COBRAML_GEMM_CUH
#define COBRAML_GEMM_CUH
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

void tn() {
    using namespace cute;

    make_tma_copy_A_sm100();

}

#endif //COBRAML_GEMM_CUH