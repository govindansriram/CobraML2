#pragma once
#include <cute/config.hpp>

namespace cobraml{


template <typename DType>
__device__ DType warp_max(DType val) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

}