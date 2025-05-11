//
// Created by sriram on 4/26/25.
//


#include <iostream>
#include <numeric>

#include "cuda_helpers.h"
#include "cuda_math.h"
#include "cuda_device_helpers.cuh"

namespace cobraml::core {
    template<typename T>
    __global__ void gemv_naive(
        const T *matrix,
        const T *vector,
        T *dest,
        const T alpha,
        const T beta,
        const size_t rows,
        const size_t columns,
        const size_t row_stride) {
        unsigned int column{blockIdx.x * blockDim.x + threadIdx.x};

        if (column < rows) {
            T reduction{0};
            for (int i{0}; i < columns; ++i) {
                reduction += matrix[row_stride * column + i] * vector[i];
            }

            dest[column] = reduction * alpha + dest[column] * beta;
        }
    }

    template<typename T, size_t BLOCK_TILE_SIZE_X>
    __global__ void gemv_1d_thread_tile(
        const T *matrix,
        const T *vector,
        T *dest,
        const T alpha,
        const T beta,
        const size_t rows,
        const size_t columns,
        const size_t row_stride) {
        __shared__ T VECTOR_TILE[BLOCK_TILE_SIZE_X];

        const uint iters{ceil_div(columns, BLOCK_TILE_SIZE_X)};
        const size_t global_row{threadIdx.x + blockIdx.x * BLOCK_TILE_SIZE_X};

        T running_sum{0};

        for (size_t iter{0}; iter < iters; ++iter) {
            // load to shared memory with coalesced memory access
            const size_t load_pos_v{iter * BLOCK_TILE_SIZE_X + threadIdx.x};
            if (load_pos_v < columns)
                VECTOR_TILE[threadIdx.x] = vector[load_pos_v];
            else
                VECTOR_TILE[threadIdx.x] = static_cast<T>(0);
            __syncthreads();

            for (size_t k{0}; k < BLOCK_TILE_SIZE_X; ++k) {
                // every thread would be accessing the same segment of the VECTOR_TILE which leads to a broadcast

                // however when accessing the matrix memory accesses wont be coalesced
                const size_t load_pos_m{row_stride * global_row + iter * BLOCK_TILE_SIZE_X + k};
                running_sum += ((global_row < rows) && (iter * BLOCK_TILE_SIZE_X + k < columns)
                                    ? matrix[load_pos_m]
                                    : static_cast<T>(0)) * VECTOR_TILE[k];
            }
            __syncthreads();
        }

        running_sum *= alpha;

        if (global_row < rows)
            dest[global_row] = dest[global_row] * beta + running_sum;
    }

    template<typename T, size_t WARP_SIZE = 32>
    __global__ void gemv_memory_coalescing(
        const T *matrix,
        const T *vector,
        T *dest,
        const T alpha,
        const T beta,
        const size_t columns,
        const size_t row_stride) {

        __shared__ T VECTOR_TILE[WARP_SIZE];
        const size_t global_row{blockIdx.x};
        const size_t iters{ceil_div(columns, WARP_SIZE)};

        T partial{0};

        // coalesced memory access every thread accesses data in iterations of 32 allowing access to be
        // coalesced
        for (size_t iter{0}; iter < iters; ++iter) {
            // load to shared memory with coalesced memory access
            const size_t load_pos_v{iter * WARP_SIZE + threadIdx.x};
            const size_t load_pos_m{row_stride * global_row + WARP_SIZE * iter + threadIdx.x};

            if (load_pos_v < columns) partial += vector[load_pos_v] * matrix[load_pos_m];
        }

        VECTOR_TILE[threadIdx.x] = partial;
        __syncthreads();

        if (threadIdx.x == 0) {
            T total{0};
#pragma unroll
            for (size_t i{0}; i < WARP_SIZE; ++i)
                total += VECTOR_TILE[i];

            dest[global_row] = dest[global_row] * beta + total * alpha;
        }
    }


    template<typename T>
    void gemv_dispatch(
        const T *matrix,
        const T *vector,
        T *dest,
        const T alpha,
        const T beta,
        const size_t rows,
        const size_t columns,
        const size_t row_stride) {
#ifdef BENCHMARK

        switch (func_pos) {
            case 0: {
                constexpr dim3 block_dim{16 * 16};
                const dim3 grid_dim{ceil_div(rows, block_dim.x)};

                gemv_naive<T><<<grid_dim, block_dim>>>(
                    matrix,
                    vector,
                    dest,
                    alpha,
                    beta,
                    rows,
                    columns,
                    row_stride);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 1: {
                constexpr uint BLOCK_TILE_SIZE_X{16 * 16};
                constexpr dim3 block_dim{BLOCK_TILE_SIZE_X};
                const dim3 grid_dim{ceil_div(rows, block_dim.x)};

                gemv_1d_thread_tile<T, BLOCK_TILE_SIZE_X><<<grid_dim, block_dim>>>(
                    matrix,
                    vector,
                    dest,
                    alpha,
                    beta,
                    rows,
                    columns,
                    row_stride);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 2: {
                constexpr uint WARP_SIZE{32};
                constexpr dim3 block_dim{WARP_SIZE};
                const dim3 grid_dim{static_cast<uint>(rows)};

                gemv_memory_coalescing<T, WARP_SIZE><<<grid_dim, block_dim>>>(
                    matrix,
                    vector,
                    dest,
                    alpha,
                    beta,
                    columns,
                    row_stride);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            default: {
                throw std::runtime_error("invalid function provided to gemv");
            }
        }
#else
        constexpr uint WARP_SIZE{32};
        constexpr dim3 block_dim{WARP_SIZE};
        const dim3 grid_dim{static_cast<uint>(rows)};

        gemv_memory_coalescing<T, WARP_SIZE><<<grid_dim, block_dim>>>(
            matrix,
            vector,
            dest,
            alpha,
            beta,
            rows,
            columns,
            row_stride);
        CUDA_CHECK(cudaGetLastError());
#endif
    }

    void CudaMath::gemv(
        const void *matrix,
        const void *vector,
        void *dest,
        const void *alpha,
        const void *beta,
        const size_t rows,
        const size_t columns,
        const size_t row_stride,
        Dtype dtype) {
        switch (dtype) {
            case (INT8): {
                const auto c_mat{static_cast<const int8_t *>(matrix)};
                const auto c_vec{static_cast<const int8_t *>(vector)};
                auto c_dest{static_cast<int8_t *>(dest)};

                const auto c_alpha{static_cast<const int8_t *>(alpha)};
                const auto c_beta{static_cast<const int8_t *>(beta)};

                gemv_dispatch<int8_t>(c_mat, c_vec, c_dest, *c_alpha, *c_beta, rows, columns, row_stride);
                return;
            }
            case (INT16): {
                const auto c_mat{static_cast<const int16_t *>(matrix)};
                const auto c_vec{static_cast<const int16_t *>(vector)};
                auto c_dest{static_cast<int16_t *>(dest)};

                const auto c_alpha{static_cast<const int16_t *>(alpha)};
                const auto c_beta{static_cast<const int16_t *>(beta)};

                gemv_dispatch<int16_t>(c_mat, c_vec, c_dest, *c_alpha, *c_beta, rows, columns, row_stride);
                return;
            }
            case (INT32): {
                const auto c_mat{static_cast<const int32_t *>(matrix)};
                const auto c_vec{static_cast<const int32_t *>(vector)};
                auto c_dest{static_cast<int32_t *>(dest)};

                const auto c_alpha{static_cast<const int32_t *>(alpha)};
                const auto c_beta{static_cast<const int32_t *>(beta)};

                gemv_dispatch<int32_t>(c_mat, c_vec, c_dest, *c_alpha, *c_beta, rows, columns, row_stride);
                return;
            }
            case (INT64): {
                const auto c_mat{static_cast<const int64_t *>(matrix)};
                const auto c_vec{static_cast<const int64_t *>(vector)};
                auto c_dest{static_cast<int64_t *>(dest)};

                const auto c_alpha{static_cast<const int64_t *>(alpha)};
                const auto c_beta{static_cast<const int64_t *>(beta)};

                gemv_dispatch<int64_t>(c_mat, c_vec, c_dest, *c_alpha, *c_beta, rows, columns, row_stride);
                return;
            }
            case (FLOAT32): {
                const auto c_mat{static_cast<const float *>(matrix)};
                const auto c_vec{static_cast<const float *>(vector)};
                auto c_dest{static_cast<float *>(dest)};

                const auto c_alpha{static_cast<const float *>(alpha)};
                const auto c_beta{static_cast<const float *>(beta)};

                gemv_dispatch<float>(c_mat, c_vec, c_dest, *c_alpha, *c_beta, rows, columns, row_stride);
                return;
            }
            case (FLOAT64): {
                const auto c_mat{static_cast<const double *>(matrix)};
                const auto c_vec{static_cast<const double *>(vector)};
                auto c_dest{static_cast<double *>(dest)};

                const auto c_alpha{static_cast<const double *>(alpha)};
                const auto c_beta{static_cast<const double *>(beta)};

                gemv_dispatch<double>(c_mat, c_vec, c_dest, *c_alpha, *c_beta, rows, columns, row_stride);
                return;
            }
            default: {
                throw std::runtime_error("invalid dtype provided to gemv");
            }
        }
    }
}
