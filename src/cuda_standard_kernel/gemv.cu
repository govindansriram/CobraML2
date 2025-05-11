//
// Created by sriram on 4/26/25.
//


#include <iostream>
#include "cuda_helpers.h"
#include "cuda_math.h"

namespace cobraml::core {

    template <typename T>
    __global__ void gemv_naive(
        const T *matrix,
        const T *vector,
        T * dest,
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

    template<typename T>
    void gemv_dispatch(
        const T * matrix,
        const T * vector,
        T * dest,
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
            default: {
                throw std::runtime_error("invalid function provided to gemv");
            }
        }
#else
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


