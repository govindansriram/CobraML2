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

            dest[column] = reduction;
        }
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

        constexpr dim3 block_dim{16 * 16};
        const dim3 grid_dim{
            static_cast<unsigned int>(std::ceil(static_cast<float>(rows) / static_cast<float>(block_dim.x)))
        };

        gemv_naive<float><<<grid_dim, block_dim>>>(
            static_cast<const float *>(matrix),
            static_cast<const float *>(vector),
            static_cast<float *>(dest),
            1.f,
            1.f,
            rows,
            columns,
            row_stride);

        CUDA_CHECK(cudaGetLastError());
    }

}


