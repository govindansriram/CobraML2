#include "cuda_helpers.h"
#include "cuda_math.h"

namespace cobraml::core {
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    __global__ void eq_reduce_naive(
        const int *in_data,
        int *out_data,
        const size_t total_elements) {
        extern __shared__ int TILE[];
        const size_t current_pos = threadIdx.x + (blockDim.x * blockIdx.x * 2);
        const size_t next_pos = threadIdx.x + blockDim.x * (blockIdx.x * 2 + 1);
        const size_t tid{threadIdx.x};

        TILE[tid] = current_pos < total_elements ? in_data[current_pos] : 1;
        TILE[tid] &= next_pos < total_elements ? in_data[next_pos] : 1;

        for (size_t jump{1}; jump < blockDim.x; jump *= 2) {
            __syncthreads();
            if (current_pos % (2 * jump) == 0)
                TILE[tid] = TILE[tid] & TILE[tid + jump];
        }

        if (threadIdx.x == 0) {
            out_data[blockIdx.x] = TILE[0];
        }
    }


    __global__ void eq_reduce_naive2(
        const int *in_data,
        int *out_data,
        const size_t total_elements) {
        extern __shared__ int TILE[];

        const size_t current_pos = threadIdx.x + (blockDim.x * blockIdx.x * 2);
        const size_t next_pos = threadIdx.x + blockDim.x * (blockIdx.x * 2 + 1);
        const size_t tid{threadIdx.x};

        TILE[tid] = current_pos < total_elements ? in_data[current_pos] : 1;
        TILE[tid] &= next_pos < total_elements ? in_data[next_pos] : 1;

        for (int jump{1}; jump < blockDim.x; jump *= 2) {
            __syncthreads();

            // most threads will now follow the same path, reducing warp divergence
            const size_t index{2 * jump * tid};
            // increases bank conflict after jump is 16 becasue 32 is added to the index which is the same bank in shared memory
            if (index < blockDim.x)
                TILE[index] = TILE[index] & TILE[index + jump];
        }

        if (threadIdx.x == 0) out_data[blockIdx.x] = TILE[0];
    }

    //coalesced with minimal bank conflicts
    __global__ void eq_reduce_naive3(
        const int *in_data,
        int *out_data,
        const size_t total_elements) {
        extern __shared__ int TILE[];
        const size_t current_pos = threadIdx.x + (blockDim.x * blockIdx.x * 2);
        const size_t next_pos = threadIdx.x + blockDim.x * (blockIdx.x * 2 + 1);
        const size_t tid{threadIdx.x};

        TILE[tid] = current_pos < total_elements ? in_data[current_pos] : 1;
        TILE[tid] &= next_pos < total_elements ? in_data[next_pos] : 1;

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            __syncthreads();
            if (tid < s) TILE[tid] &= TILE[tid + s];
        }

        if (threadIdx.x == 0) out_data[blockIdx.x] = TILE[0];
    }

    // // not coalesced
    // __global__ void eq_reduce_naive4(
    //     const int *in_data,
    //     int *out_data,
    //     const size_t total_elements) {
    //     extern __shared__ int TILE[];
    //     const unsigned int pos = threadIdx.x + (blockDim.x * blockIdx.x);
    //     const unsigned int tid{threadIdx.x};
    //
    //     if (pos < total_elements) {
    //         TILE[tid] = in_data[pos];
    //     } else TILE[tid] = 1;
    //
    //     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    //         __syncthreads();
    //         // add first and last element to avoid bank conflicts
    //         if (tid < s) TILE[tid] &= TILE[(s * 2) - (tid + 1)];
    //     }
    //
    //     if (threadIdx.x == 0) {
    //         out_data[blockIdx.x] = TILE[0];
    //     }
    // }

    // volatile is required this removes the need for syncs since the last warp does not need syncs
    __device__ void warp_reduce(volatile int* sdata, const size_t tid) {
        sdata[tid] &= sdata[tid + 32];
        sdata[tid] &= sdata[tid + 16];
        sdata[tid] &= sdata[tid + 8];
        sdata[tid] &= sdata[tid + 4];
        sdata[tid] &= sdata[tid + 2];
        sdata[tid] &= sdata[tid + 1];
    }

    //coalesced with minimal bank conflicts
    __global__ void eq_reduce_naive4(
        const int *in_data,
        int *out_data,
        const size_t total_elements) {
        extern __shared__ int TILE[];
        const size_t current_pos = threadIdx.x + (blockDim.x * blockIdx.x * 2);
        const size_t next_pos = threadIdx.x + blockDim.x * (blockIdx.x * 2 + 1);
        const size_t tid{threadIdx.x};

        TILE[tid] = current_pos < total_elements ? in_data[current_pos] : 1;
        TILE[tid] &= next_pos < total_elements ? in_data[next_pos] : 1;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) TILE[tid] &= TILE[tid + s];
            __syncthreads();
        }

        if (tid < 32) warp_reduce(TILE, tid);

        if (threadIdx.x == 0) out_data[blockIdx.x] = TILE[0];
    }

    // Adding this function to help with unrolling and adding the Template
    template <size_t block_size>
    __device__ void bsf_warp_reduce(volatile int* sdata, const size_t tid){
        // block size friendly
        if(block_size >= 64) sdata[tid] &= sdata[tid + 32];
        if(block_size >= 32) sdata[tid] &= sdata[tid + 16];
        if(block_size >= 16) sdata[tid] &= sdata[tid + 8];
        if(block_size >= 8) sdata[tid] &= sdata[tid + 4];
        if(block_size >= 4) sdata[tid] &= sdata[tid + 2];
        if(block_size >= 2) sdata[tid] &= sdata[tid + 1];
    }

    //coalesced with minimal bank conflicts
    template<size_t block_size>
    __global__ void eq_reduce_naive5(
        const int *in_data,
        int *out_data,
        const size_t total_elements) {

        extern __shared__ int TILE[];
        const size_t current_pos = threadIdx.x + (blockDim.x * blockIdx.x * 2);
        const size_t next_pos = threadIdx.x + blockDim.x * (blockIdx.x * 2 + 1);
        const size_t tid{threadIdx.x};

        TILE[tid] = current_pos < total_elements ? in_data[current_pos] : 1;
        TILE[tid] &= next_pos < total_elements ? in_data[next_pos] : 1;
        __syncthreads();

        if (block_size >= 512) {
            if (tid < 256) {
                TILE[tid] &= TILE[tid + 256];
            }
            __syncthreads();
        }

        if (block_size >= 256) {
            if (tid < 128) {
                TILE[tid] &= TILE[tid + 128];
            }
            __syncthreads();
        }

        if (block_size >= 128) {
            if (tid < 64) {
                TILE[tid] &= TILE[tid + 64];
            }
            __syncthreads();
        }

        if (tid < 32) bsf_warp_reduce<block_size>(TILE, tid);
        if (tid == 0) out_data[blockIdx.x] = TILE[0];
    }

    // coalesced with minimal bank conflicts
    // cascaded over grid size
    // TODO: test shortening blocks by larger factors that are multiple of 2 maybe 4 or 6 etc
    template<size_t block_size>
    __global__ void eq_reduce_naive6(
        const int *in_data,
        int *out_data,
        const size_t total_elements) {

        extern __shared__ int TILE[];
        size_t current_pos{threadIdx.x + (blockDim.x * blockIdx.x * 2)};
        const size_t grid_size{threadIdx.x + (blockDim.x * 2) * gridDim.x};
        const size_t tid{threadIdx.x};

        TILE[tid] = 1;
        while (current_pos < total_elements) {
            TILE[tid] &= in_data[current_pos];
            current_pos += grid_size;
        }
        __syncthreads();

        if (block_size >= 512) {
            if (tid < 256) {
                TILE[tid] &= TILE[tid + 256];
            }
            __syncthreads();
        }

        if (block_size >= 256) {
            if (tid < 128) {
                TILE[tid] &= TILE[tid + 128];
            }
            __syncthreads();
        }

        if (block_size >= 128) {
            if (tid < 64) {
                TILE[tid] &= TILE[tid + 64];
            }
            __syncthreads();
        }

        if (tid < 32) bsf_warp_reduce<block_size>(TILE, tid);
        if (tid == 0) out_data[blockIdx.x] = TILE[0];
    }

    template<typename DataType>
    __global__ void eq_mask(
        const DataType *arr_one,
        const DataType *arr_two,
        int *out,
        const size_t total) {
        size_t pos = threadIdx.x + (blockDim.x * blockIdx.x);
        if (pos < total)
            out[pos] = arr_one[pos] == arr_two[pos];
    }

    void eq_dispatcher(
        int *in,
        int *out,
        const size_t block_size,
        const size_t grid_size,
        const size_t s_data_size,
        const size_t total_elements) {
#ifdef BENCHMARK
        switch (func_pos) {
            case 0: {
                eq_reduce_naive<<<grid_size, block_size, s_data_size>>>(
                    in,
                    out,
                    total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 1: {
                eq_reduce_naive2<<<grid_size, block_size, s_data_size>>>(
                    in,
                    out,
                    total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 2: {
                eq_reduce_naive3<<<grid_size, block_size, s_data_size>>>(
                    in,
                    out,
                    total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 3: {
                eq_reduce_naive4<<<grid_size, block_size, s_data_size>>>(
                    in,
                    out,
                    total_elements);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 4: {
                eq_reduce_naive5<256><<<grid_size, block_size, s_data_size>>>(
                    in,
                    out,
                    total_elements);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 5: {
                eq_reduce_naive6<256><<<grid_size, block_size, s_data_size>>>(
                    in,
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
        eq_reduce_naive6<256><<<grid_size, block_size, s_data_size>>>(
            in,
            out,
            total_elements);

        CUDA_CHECK(cudaGetLastError());
#endif
    }

    template<typename DataType>
    void eq_wrapper(
        const DataType *arr_one,
        const DataType *arr_two,
        int *out,
        size_t total_elements) {
        constexpr unsigned int block_size{TILE_WIDTH * TILE_WIDTH};

        size_t blocks{calculate_dim(total_elements, block_size)};
        eq_mask<DataType><<<blocks, block_size>>>(arr_one, arr_two, out, total_elements);
        CUDA_CHECK(cudaGetLastError());

        int *in;
        CUDA_CHECK(cudaMalloc(&in, blocks * 4));
        int *d_in{out};
        int *d_out{in};

        while (total_elements > 1) {
            blocks = (blocks + 1) / 2;

            eq_dispatcher(
                d_in,
                d_out,
                block_size,
                blocks,
                block_size * 4,
                total_elements);

            total_elements = blocks;
            blocks = calculate_dim(total_elements, block_size);

            int *temp = d_in;
            d_in = d_out;
            d_out = temp;
        }

        CUDA_CHECK(cudaMemcpy(out, d_in, 4, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(in));
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
            case INT8: {
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
