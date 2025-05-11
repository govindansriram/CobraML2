//
// Created by sriram on 5/10/25.
//

#ifndef CUDA_DEVICE_HELPERS_CUH
#define CUDA_DEVICE_HELPERS_CUH

__device__ __host__ __forceinline__ uint ceil_div(const uint a, const uint b) {
    return (a + b - 1) / b;
}

#endif //CUDA_DEVICE_HELPERS_CUH
