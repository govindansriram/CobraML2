#pragma once
#include <cuda/std/functional>
#include <cute/config.hpp>

namespace cobraml {

template <typename DType> __device__ DType warp_max(DType val) {
  CUTE_UNROLL
  for (int offset = 16; offset > 0; offset /= 2)
    val = cuda::std::max(val, __shfl_xor_sync(0xffffffff, val, offset));
  return val;
}

template <typename DType> __device__ DType warp_sum(DType val) {

  CUTE_UNROLL
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

} // namespace cobraml