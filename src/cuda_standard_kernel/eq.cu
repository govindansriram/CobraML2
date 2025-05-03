#include "cuda_helpers.h"
#include "cuda_math.h"

namespace cobraml::core {

    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    __global__ void eq_reduce_naive(
        int * data,
        const size_t total_elements) {

        extern __shared__ int TILE[];
        const size_t pos = threadIdx.x + (blockDim.x * blockIdx.x);
        const size_t tid{threadIdx.x};

        if (pos < total_elements) {
            TILE[tid] = data[pos];
            data[pos] = 1;
        }else TILE[tid] = 1;

        for (size_t jump{1}; jump < blockDim.x; jump *= 2) {
            __syncthreads();
            if (pos % (2 * jump) == 0)
                TILE[tid] = TILE[tid] & TILE[tid + jump];
        }

        if (threadIdx.x == 0) data[blockIdx.x] = TILE[0];
    }


    __global__ void eq_reduce_naive2(
        int * data,
        const size_t total_elements) {

        extern __shared__ int TILE[];
        const unsigned int pos = threadIdx.x + (blockDim.x * blockIdx.x);
        const unsigned int tid{threadIdx.x};

        if (pos < total_elements) {
            TILE[tid] = data[pos];
            data[pos] = 1;
        }else TILE[tid] = 1;

        for (int jump{1}; jump < blockDim.x; jump *= 2) {
            __syncthreads();

            // most threads will now follow the same path, reducing warp divergence
            const unsigned int index{2 * jump * tid}; // increases bank conflict after jump is 16 becasue 32 is added to the index which is the same bank in shared memory
            if (index < blockDim.x)
                TILE[index] = TILE[index] & TILE[index + jump];
        }

        if (threadIdx.x == 0) data[blockIdx.x] = TILE[0];
    }

    __global__ void eq_reduce_naive3(
    int * data,
    const size_t total_elements) {

        extern __shared__ int TILE[];
        const unsigned int pos = threadIdx.x + (blockDim.x * blockIdx.x);
        const unsigned int tid{threadIdx.x};

        if (pos < total_elements) {
            TILE[tid] = data[pos];
            data[pos] = 1;
        }else TILE[tid] = 1;

        for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
            if (tid < s) {
                TILE[tid] &= TILE[tid + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) data[blockIdx.x] = TILE[0];
    }

    template<typename DataType>
    __global__ void eq_mask(
        const DataType * arr_one,
        const DataType * arr_two,
        int * out,
        const size_t total) {
        size_t pos = threadIdx.x + (blockDim.x * blockIdx.x);
        if (pos < total)
            out[pos] = arr_one[pos] == arr_two[pos];
    }

    void eq_dispatcher(
        int * out,
        const dim3 block_size,
        const dim3 grid_size,
        const size_t s_data_size,
        const size_t total_elements) {
#ifdef BENCHMARK
        switch (func_pos) {
            case 0: {
                eq_reduce_naive<<<grid_size, block_size, s_data_size>>>(
                    out,
                    total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 1: {
                eq_reduce_naive2<<<grid_size, block_size, s_data_size>>>(
                            out,
                            total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 2: {
                eq_reduce_naive3<<<grid_size, block_size, s_data_size>>>(
                            out,
                            total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            default: {
                throw std::runtime_error("invalid function requested for eq operator");
            }
        }
#else
        eq_reduce_naive3<<<grid_size, block_size, s_data_size>>>(
                            out,
                            total_elements);

        CUDA_CHECK(cudaGetLastError());
#endif
    }

    template<typename DataType>
    void eq_wrapper(
        const DataType * arr_one,
        const DataType * arr_two,
        int * out,
        const size_t total_elements) {

        constexpr unsigned int shared_memory_count{TILE_WIDTH * TILE_WIDTH};
        dim3 block_size{shared_memory_count};

        unsigned int sz{calculate_dim(total_elements, shared_memory_count)};
        eq_mask<DataType><<<sz, block_size>>>(arr_one, arr_two, out, total_elements);
        CUDA_CHECK(cudaGetLastError());

        while (sz > 1) {
            const dim3 grid_size{sz};
            eq_dispatcher(
                out,
                block_size,
                grid_size,
                shared_memory_count * 4,
                total_elements);
            sz /= 2;
            sz = static_cast<unsigned int>(std::pow(2,std::ceil(std::log(sz) / std::log(2))));
        }
    }

    void CudaMath::equals(
            const void *tensor_1,
            const void *tensor_2,
            int *result,
            const size_t *tensor_shape,
            const size_t *tensor_stride,
            const Dtype dtype) {

        const size_t total_elements{calculate_total_elements(tensor_shape, tensor_stride)};

        switch (dtype) {
            case INT8:{
                const auto a1{static_cast<const int8_t *>(tensor_1)};
                const auto a2{static_cast<const int8_t *>(tensor_2)};
                eq_wrapper<int8_t>(a1, a2, result, total_elements);
                return;
            }
            case INT16: {
                const auto a1{static_cast<const int16_t *>(tensor_1)};
                const auto a2{static_cast<const int16_t *>(tensor_2)};
                eq_wrapper<int16_t>(a1, a2, result, total_elements);
                return;
            }
            case INT32: {
                const auto a1{static_cast<const int32_t *>(tensor_1)};
                const auto a2{static_cast<const int32_t *>(tensor_2)};
                eq_wrapper<int32_t>(a1, a2, result, total_elements);
            }
                return;
            case INT64: {
                const auto a1{static_cast<const int64_t *>(tensor_1)};
                const auto a2{static_cast<const int64_t *>(tensor_2)};
                eq_wrapper<int64_t>(a1, a2, result, total_elements);
                return;
            }
            case FLOAT32: {
                const auto a1{static_cast<const float *>(tensor_1)};
                const auto a2{static_cast<const float *>(tensor_2)};
                eq_wrapper<float>(a1, a2, result, total_elements);
                return;
            }
            case FLOAT64: {
                const auto a1{static_cast<const double *>(tensor_1)};
                const auto a2{static_cast<const double *>(tensor_2)};
                eq_wrapper<double>(a1, a2, result, total_elements);
                return;
            }
            case INVALID: {
                throw std::runtime_error("invalid datatype provided for equals");
            }
        }
    }
}
