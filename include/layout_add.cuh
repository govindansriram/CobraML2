//
// Created by sriram on 6/28/25.
//

#ifndef LAYOUT_ADD_CUH
#define LAYOUT_ADD_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cute/tensor.hpp>

template<
    typename A_TYPE,
    typename B_TYPE,
    typename A_LAYOUT,
    typename B_LAYOUT
>
__global__ void tensor_add(
    const A_TYPE *gmem_A,
    const B_TYPE *gmem_B,
    const A_LAYOUT layout_A,
    const B_LAYOUT layout_B
) {

    if (threadIdx.x == 0) {
        printf("yooooo \n");
    }

}

#endif //LAYOUT_ADD_CUH
