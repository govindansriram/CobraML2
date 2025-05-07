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
                accum += matrix_one[row_stride_one * row + i] * matrix_two[row_stride_two * i + column];

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

        const size_t row_block{blockIdx.y};
        const size_t row_thread{threadIdx.y};

        const size_t column_block{blockIdx.x};
        const size_t column_thread{threadIdx.x};

        const size_t row{row_block * blockDim.y + row_thread};
        const size_t column{column_block * blockDim.x + column_thread};

        T dummy{0};

        // ensure data is written within a boundary with minimal divergence overhead
        T *dest = (row < mat_one_rows && column < mat_two_columns)
                      ? &matrix_dest[row_stride_dest * row + column]
                      : &dummy;

        T accumulation{0};

        const size_t limit{(shared + (TILE_WIDTH - 1)) / TILE_WIDTH};
        for (size_t i{0}; i < limit; ++i) {
            // shift right
            // boundary mask for tiles over matrix boundaries
            TILE_BLOCK_ONE[row_thread][column_thread] =
                    (TILE_WIDTH * i + column_thread < shared && row < mat_one_rows)
                        ? matrix_one[row_stride_one * row + (TILE_WIDTH * i + column_thread)]
                        : static_cast<T>(0);

            // shift down
            // boundary mask for tiles over matrix boundaries
            TILE_BLOCK_TWO[row_thread][column_thread] =
                    (TILE_WIDTH * i + row_thread < shared && column < mat_two_columns)
                        ? matrix_two[row_stride_two * (TILE_WIDTH * i + row_thread) + column]
                        : static_cast<T>(0);

            __syncthreads();

            for (size_t j{0}; j < TILE_WIDTH; ++j) {
                accumulation += TILE_BLOCK_ONE[row_thread][j] * TILE_BLOCK_TWO[j][column_thread];
            }

            __syncthreads();
        }

        *dest *= beta;
        *dest += alpha * accumulation;
    }

    __device__ __forceinline__ uint ceil_div(const uint a, const uint b) {
        return (a + b - 1) / b;
    }

    template<typename T, uint BLOCK_TILE_SIZE_X, uint BLOCK_TILE_SIZE_Y, uint BLOCK_TILE_SIZE_K,
             uint THREADS_PER_BLOCK>
    __device__ void load_data_to_shared_memory(
        const T * matrix_one,
        const T * matrix_two,
        const size_t stride_one,
        const size_t stride_two,
        T one_shared[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K],
        T two_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const uint iteration,
        const uint linear_idx) {

        /**
           we divide the size of the shared matrix by the amount of threads per block, this tells us how
           many pieces of data each thread has to fetch from the global matrices to fill out the shared
           block
        */
        const uint shared_one_iters{ceil_div(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)};

        for (uint load_idx{0}; load_idx < shared_one_iters; ++load_idx) {
            const size_t shared_block_one_row{
                (linear_idx + load_idx * THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_K};

            const size_t shared_block_one_column{
                (linear_idx + load_idx * THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_K};

            const size_t matrix_one_row{shared_block_one_row + blockIdx.y * BLOCK_TILE_SIZE_Y};
            const size_t matrix_one_column{iteration * BLOCK_TILE_SIZE_K + shared_block_one_column};

            T val{static_cast<T>(0)};
            if (matrix_one_row < mat_one_rows && matrix_one_column < shared)
                val = matrix_one[matrix_one_row * stride_one + matrix_one_column];

            one_shared[shared_block_one_row][shared_block_one_column] = val;
        }

        const uint shared_two_iters{ceil_div(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)};

        for (uint load_idx{0}; load_idx < shared_two_iters; ++load_idx) {
            const size_t shared_block_two_row{
                (linear_idx + load_idx * THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_X};

            const size_t shared_block_two_column{
                (linear_idx + load_idx * THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_X};

            const size_t matrix_two_row{iteration * BLOCK_TILE_SIZE_K + shared_block_two_row};
            const size_t matrix_two_column{blockIdx.x * BLOCK_TILE_SIZE_X + shared_block_two_column};

            T val{static_cast<T>(0)};
            if (matrix_two_row < shared && matrix_two_column < mat_two_columns)
                val = matrix_two[matrix_two_row * stride_two + matrix_two_column];

            two_shared[shared_block_two_row][shared_block_two_column] = val;
        }

    }

    template <typename T, uint BLOCK_TILE_SIZE_X, uint BLOCK_TILE_SIZE_Y, uint BLOCK_TILE_SIZE_K>
    __global__ void gemm_tiled_2(
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

        // Cache a tile of A and B in shared memory for data reuse.
        __shared__ T mat_one_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        // Compute the row and column of dest that this thread is responsible for.

        const uint dest_column{threadIdx.x + blockDim.x * blockIdx.x};
        const uint dest_row{threadIdx.y + blockDim.y * blockIdx.y};

        const uint total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        constexpr uint threads_per_block{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};

        T running_sum{static_cast<T>(0)};

        for (uint iter{0}; iter < total_iters; ++iter) {
            // load into shared memory

            __syncthreads();
        }

    }

    // Used implementation by https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
    // double buffering cuda
    // undestand cache management

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

        switch (func_pos) {
            case 0: {
                constexpr dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
                const dim3 grid_dim(
                    static_cast<size_t>(
                        std::ceil(static_cast<float>(mat_two_columns) / static_cast<float>(TILE_WIDTH))),
                    static_cast<size_t>(std::ceil(static_cast<float>(mat_one_rows) / static_cast<float>(TILE_WIDTH)))
                );
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
                return;
            }
            case 2: {
                constexpr uint BLOCK_TILE_SIZE_X{32};
                constexpr uint BLOCK_TILE_SIZE_Y{32};
                constexpr uint BLOCK_TILE_SIZE_K{32};

                constexpr uint TOTAL_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};

                // ensure shared x and y block are divisible by total threads
                static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % TOTAL_THREADS == 0);
                static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % TOTAL_THREADS == 0);

                constexpr dim3 block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1};

                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };



                CUDA_CHECK(cudaGetLastError());
            }
            default: {
                throw std::runtime_error("invalid function provided");
            }
        }
#else
        constexpr dim3 block_dim(TILE_WIDTH, TILE_WIDTH);

        const dim3 grid_dim(
            calculate_dim(mat_two_columns, TILE_WIDTH),
            calculate_dim(mat_one_rows, TILE_WIDTH)
        );

        gemm_tiled<T><<<grid_dim, block_dim>>>(
            matrix_one,
            matrix_two,
            matrix_dest,
            alpha, beta,
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
