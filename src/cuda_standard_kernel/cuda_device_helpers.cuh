//
// Created by sriram on 5/10/25.
//

#ifndef CUDA_DEVICE_HELPERS_CUH
#define CUDA_DEVICE_HELPERS_CUH

__device__ __host__ __forceinline__ constexpr uint ceil_div(const uint a, const uint b) {
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
        const size_t c_warp{tidx / WARP_SIZE};
        const size_t c_t_pos{tidx % WARP_SIZE};

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

#define CUDA_EXPR_BARRIER(block_x_id, block_y_id, thread_x_id, thread_y_id, expression)      \
    {                                                                                        \
        bool cond1{thread_x_id == threadIdx.x && thread_y_id == threadIdx.y};                \
        bool cond2{block_x_id == blockIdx.x && block_y_id == block_y_id};                    \
        if (cond1 && cond2) {                                                                \
            expression                                                                       \
        }                                                                                    \
    }                                                                                        \


#define CUDA_EXPR_BARRIER_0(expression) CUDA_EXPR_BARRIER(0, 0, 0, 0, expression)


template<
    typename T,
    size_t COLUMN_SIZE>
 __device__ void print_vector(T vector[COLUMN_SIZE]) {
    printf("[");
    for (size_t i{0}; i < COLUMN_SIZE; ++i) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) printf("%f ", vector[i]);
        else if constexpr (std::is_same_v<T, int64_t>) printf("%lld ", vector[i]);
        else printf("%d ", vector[i]);
    }
    printf("]");
}

template<
    typename T,
    size_t ROW_SIZE,
    size_t COLUMN_SIZE>
__device__ void print_matrix(
    T matrix[ROW_SIZE][COLUMN_SIZE]) {
    for (size_t i{0}; i < ROW_SIZE; ++i) print_vector<T, COLUMN_SIZE>(matrix[i]);
}

#endif //CUDA_DEVICE_HELPERS_CUH
