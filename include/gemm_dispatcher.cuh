//
// Created by sriram on 8/11/25.
//

#ifndef COBRAML_GEMM_DISPATCHER_CUH
#define COBRAML_GEMM_DISPATCHER_CUH

#include <cute/layout.hpp>
#include "helpers/utils.h"

namespace cobraml {
    using namespace cute;
    template<
        typename AType,
        typename BType,
        typename CType,
        typename MatrixShape,
        typename MatrixMajor
    >
    void gemm(
        AType * a,
        BType * b,
        CType * c,
        MatrixShape matrix_shape,
        MatrixMajor matrix_major
    ) {

    }

}

#endif //COBRAML_GEMM_DISPATCHER_CUH
