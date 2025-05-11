//
// Created by sriram on 5/10/25.
//

#ifndef CUDA_DEVICE_HELPERS_CUH
#define CUDA_DEVICE_HELPERS_CUH

__device__ __host__ __forceinline__ uint ceil_div(const uint a, const uint b) {
    return (a + b - 1) / b;
}

template<typename T, size_t WARP_SIZE=32>
__device__ __forceinline__ T warp_reduction(T value) {

    for (size_t offset{WARP_SIZE / 2}; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);

    return value;
}

#endif //CUDA_DEVICE_HELPERS_CUH
