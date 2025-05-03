#include "cuda_math.h"
#include "cuda_helpers.h"

namespace cobraml::core {

    template<typename T>
    __global__ void gemm_naive(
        const T *matrix_one,
        const T *matrix_two,
        T *matrix_dest,
        const T alpha,
        const T beta,
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const size_t row_stride_one,
        const size_t row_stride_two,
        const size_t row_stride_dest) {
        const size_t row{blockDim.y * blockIdx.y + threadIdx.y};
        const size_t column{blockDim.x * blockIdx.x + threadIdx.x};

        if (row < mat_one_rows && column < mat_two_columns) {
            matrix_dest[row * row_stride_dest + column] *= beta;

            T accum{0};
            for (size_t i{0}; i < shared; ++i)
                accum += matrix_one[row_stride_one * row + i] + matrix_two[row_stride_two * i + column];

            accum *= alpha;
            matrix_dest[row * row_stride_dest + column] += accum;
        }
    }

    template<typename T>
    __global__ void gemm_tiled(
        const T *matrix_one,
        const T *matrix_two,
        T *matrix_dest,
        const T alpha,
        const T beta,
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const size_t row_stride_one,
        const size_t row_stride_two,
        const size_t row_stride_dest) {
        __shared__ T TILE_BLOCK_ONE[TILE_WIDTH][TILE_WIDTH]; // shared amongst all threads (in block)
        __shared__ T TILE_BLOCK_TWO[TILE_WIDTH][TILE_WIDTH]; // shared amongst all threads (in block)

        const size_t row_block{blockDim.y};
        const size_t row_thread{threadIdx.y};
        const size_t column_block{blockDim.x};
        const size_t column_thread{threadIdx.x};

        const size_t row{row_block * blockDim.y + row_thread};
        const size_t column{column_block * blockDim.x + column_thread};

        T dummy{0};

        // ensure data is written within a boundary with minimal divergence overhead
        T *dest = (row < mat_one_rows && column < mat_two_columns)
                      ? &matrix_dest[row_stride_dest * row + column]
                      : &dummy;

        T accumulation{0};

        const size_t limit{static_cast<size_t>(std::ceil(static_cast<float>(shared) / static_cast<float>(TILE_WIDTH)))};

        for (size_t i{0}; i < limit; ++i) {
            // boundary mask for tiles over matrix boundaries
            TILE_BLOCK_ONE[row_thread][column_thread] =
                    (TILE_WIDTH * i + column_thread < shared && row < mat_one_rows)
                        ? matrix_one[row_stride_one * row + (TILE_WIDTH * i + column_thread)]
                        : static_cast<T>(0);

            // boundary mask for tiles over matrix boundaries
            TILE_BLOCK_TWO[row_thread][column_thread] =
                    (TILE_WIDTH * i + row_thread < shared && column < mat_two_columns)
                        ? matrix_two[row_stride_two * (TILE_WIDTH * i + row_thread) + column]
                        : static_cast<T>(0);

            __syncthreads();

            for (size_t j{0}; j < TILE_WIDTH; ++j)
                accumulation += TILE_BLOCK_ONE[row_thread][j] * TILE_BLOCK_TWO[j][column_thread];

            __syncthreads();
        }

        *dest *= beta;
        *dest += alpha * accumulation;
    }

    template<typename T>
    static void gemm_dispatch(
        const T *matrix_one,
        const T *matrix_two,
        T *matrix_dest,
        const T alpha,
        const T beta,
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const size_t row_stride_one,
        const size_t row_stride_two,
        const size_t row_stride_dest) {
#ifdef BENCHMARK
        constexpr dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
        const dim3 grid_dim(
            static_cast<size_t>(
                std::ceil(static_cast<float>(mat_two_columns) / static_cast<float>(TILE_WIDTH))),
            static_cast<size_t>(std::ceil(static_cast<float>(mat_one_rows) / static_cast<float>(TILE_WIDTH)))
        );

        switch (func_pos) {
            case 0: {
                gemm_naive<T><<<grid_dim, block_dim>>>(
                    matrix_one,
                    matrix_two,
                    matrix_dest,
                    alpha,
                    beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 1: {
                gemm_tiled<T><<<grid_dim, block_dim>>>(
                    matrix_one,
                    matrix_two,
                    matrix_dest,
                    alpha,
                    beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            default: {
                throw std::runtime_error("invalid function provided");
            }
        }
#else
        constexpr dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
        const dim3 grid_dim(
            static_cast<size_t>(
                std::ceil(static_cast<float>(mat_two_columns) / static_cast<float>(TILE_WIDTH))),
            static_cast<size_t>(std::ceil(static_cast<float>(mat_one_rows) / static_cast<float>(TILE_WIDTH)))
        );

        gemm_tiled<T><<<grid_dim, block_dim>>>(
            matrix_one,
            matrix_two,
            matrix_dest,
            alpha,
            beta,
            mat_one_rows,
            mat_two_columns,
            shared,
            row_stride_one,
            row_stride_two,
            row_stride_dest);

        CUDA_CHECK(cudaGetLastError());
#endif
    }

    void CudaMath::gemm(
        const void *matrix_one,
        const void *matrix_two,
        void *matrix_dest,
        const void *alpha,
        const void *beta,
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const size_t row_stride_one,
        const size_t row_stride_two,
        const size_t row_stride_dest,
        const Dtype dtype) {
        switch (dtype) {
            case INT8: {
                const auto *cast_m1{static_cast<const int8_t *>(matrix_one)};
                const auto *cast_m2{static_cast<const int8_t *>(matrix_two)};
                auto *cast_dest{static_cast<int8_t *>(matrix_dest)};
                const auto cast_alpha{*static_cast<const int8_t *>(alpha)};
                const auto cast_beta{*static_cast<const int8_t *>(beta)};

                gemm_dispatch<int8_t>(
                    cast_m1,
                    cast_m2,
                    cast_dest,
                    cast_alpha,
                    cast_beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                return;
            }
            case INT16: {
                const auto *cast_m1{static_cast<const int16_t *>(matrix_one)};
                const auto *cast_m2{static_cast<const int16_t *>(matrix_two)};
                auto *cast_dest{static_cast<int16_t *>(matrix_dest)};
                const auto cast_alpha{*static_cast<const int16_t *>(alpha)};
                const auto cast_beta{*static_cast<const int16_t *>(beta)};

                gemm_dispatch<int16_t>(
                    cast_m1,
                    cast_m2,
                    cast_dest,
                    cast_alpha,
                    cast_beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                return;
            }
            case INT32: {
                const auto *cast_m1{static_cast<const int32_t *>(matrix_one)};
                const auto *cast_m2{static_cast<const int32_t *>(matrix_two)};
                auto *cast_dest{static_cast<int32_t *>(matrix_dest)};
                const auto cast_alpha{*static_cast<const int32_t *>(alpha)};
                const auto cast_beta{*static_cast<const int32_t *>(beta)};

                gemm_dispatch<int32_t>(
                    cast_m1,
                    cast_m2,
                    cast_dest,
                    cast_alpha,
                    cast_beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                return;
            }
            case INT64: {
                const auto *cast_m1{static_cast<const int64_t *>(matrix_one)};
                const auto *cast_m2{static_cast<const int64_t *>(matrix_two)};
                auto *cast_dest{static_cast<int64_t *>(matrix_dest)};
                const auto cast_alpha{*static_cast<const int64_t *>(alpha)};
                const auto cast_beta{*static_cast<const int64_t *>(beta)};

                gemm_dispatch<int64_t>(
                    cast_m1,
                    cast_m2,
                    cast_dest,
                    cast_alpha,
                    cast_beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                return;
            }

            case FLOAT32: {
                const auto *cast_m1{static_cast<const float *>(matrix_one)};
                const auto *cast_m2{static_cast<const float *>(matrix_two)};
                auto *cast_dest{static_cast<float *>(matrix_dest)};
                const auto cast_alpha{*static_cast<const float *>(alpha)};
                const auto cast_beta{*static_cast<const float *>(beta)};

                gemm_dispatch<float>(
                    cast_m1,
                    cast_m2,
                    cast_dest,
                    cast_alpha,
                    cast_beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                return;
            }
            case FLOAT64: {
                const auto *cast_m1{static_cast<const double *>(matrix_one)};
                const auto *cast_m2{static_cast<const double *>(matrix_two)};
                auto *cast_dest{static_cast<double *>(matrix_dest)};
                const auto cast_alpha{*static_cast<const double *>(alpha)};
                const auto cast_beta{*static_cast<const double *>(beta)};

                gemm_dispatch<double>(
                    cast_m1,
                    cast_m2,
                    cast_dest,
                    cast_alpha,
                    cast_beta,
                    mat_one_rows,
                    mat_two_columns,
                    shared,
                    row_stride_one,
                    row_stride_two,
                    row_stride_dest);

                return;
            }
            default: {
                throw std::runtime_error("invalid dtype provided to gemv");
            }
        }
    }
}
