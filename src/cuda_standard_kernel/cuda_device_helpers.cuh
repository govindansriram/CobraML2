//
// Created by sriram on 5/10/25.
//

#ifndef CUDA_DEVICE_HELPERS_CUH
#define CUDA_DEVICE_HELPERS_CUH

__device__ __host__ __forceinline__ uint ceil_div(const uint a, const uint b) {
    return (a + b - 1) / b;
}

template<typename T, size_t WARP_SIZE = 32>
__device__ __forceinline__ T warp_reduction(T value) {
    for (size_t offset{WARP_SIZE / 2}; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);

    return value;
}

template<typename T, size_t THREADS_PER_BLOCK, size_t WARP_SIZE = 32>
__device__ __forceinline__ void block_warp_reduction(T value, const uint tidx, T *smmem) {
    T local_sum{warp_reduction<T, WARP_SIZE>(value)};
    if (THREADS_PER_BLOCK > WARP_SIZE) {
        const uint c_warp{tidx / WARP_SIZE};
        const uint c_t_pos{tidx % WARP_SIZE};

        if (c_t_pos == 0) smmem[c_warp] = local_sum;

        __syncthreads();

        if (tidx < WARP_SIZE) {
            const uint segments = ceil_div(THREADS_PER_BLOCK, WARP_SIZE);
            local_sum = tidx < segments ? smmem[tidx] : static_cast<T>(0);
            local_sum = warp_reduction<T, WARP_SIZE>(local_sum);
            if (tidx == 0) smmem[0] = local_sum;
        }
    } else {
        if (tidx == 0) smmem[0] = local_sum;
    }
}

#endif //CUDA_DEVICE_HELPERS_CUH
