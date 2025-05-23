#include <assert.h>

#include "cuda_math.h"
#include "cuda_helpers.h"
#include "cuda_device_helpers.cuh"

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

    void gemm_cublas(
        const float *matrix_one,
        const float *matrix_two,
        float *matrix_dest,
        const float alpha,
        const float beta,
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const size_t row_stride_one,
        const size_t row_stride_two,
        const size_t row_stride_dest) {

        cublasSgemm(
            get_handle(),
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            static_cast<int>(mat_two_columns),
            static_cast<int>(mat_one_rows),
            static_cast<int>(shared),
            &alpha,
            matrix_two,
            static_cast<int>(row_stride_two),
            matrix_one,
            static_cast<int>(row_stride_one),
            &beta,
            matrix_dest,
            static_cast<int>(row_stride_dest));
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

    // __device__ __forceinline__ uint d_ceil_div(const uint a, const uint b) {
    //     return (a + b - 1) / b;
    // }

    template<typename T, uint BLOCK_TILE_SIZE_X, uint BLOCK_TILE_SIZE_Y, uint BLOCK_TILE_SIZE_K,
        uint THREADS_PER_BLOCK>
    __device__ void load_data_to_shared_memory(
        const T *matrix_one,
        const T *matrix_two,
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
         * We divide the size of the shared matrix by the amount of threads per block, this tells us how
           many pieces of data each thread has to fetch from the global matrices to fill out the shared
           block.

           if multiple pieces of data need to be fetched we achive this by looping the appropriate amount of times
           we then jump by the next prospective block size (THREADS_PER_BLOCK), by doing so each thread in the warp
           maintians memory coalescing fro mmeory sicne accesses stay sequentail and no bank conflcts are present since
           writes will also be sequential.
         */

        // required to be perfectly divisible
        const uint shared_one_iters{ceil_div(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)};
        /**
         *we can do unrolling here since realistically the amount of load iterations should be quite small
         */
#pragma unroll
        for (uint load_idx{0}; load_idx < shared_one_iters; ++load_idx) {
            // the row of the matrix one shared block
            const size_t shared_block_one_row{
                (linear_idx + load_idx * THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_K
            };

            // the column of matrix one shared block
            const size_t shared_block_one_column{
                (linear_idx + load_idx * THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_K
            };

            // the corresponding row in matrix one this should be consistent
            const size_t matrix_one_row{shared_block_one_row + blockIdx.y * BLOCK_TILE_SIZE_Y};

            // the corresponding column in matrix one (you can see that it gets shifted based on iteration)
            const size_t matrix_one_column{iteration * BLOCK_TILE_SIZE_K + shared_block_one_column};

            // load a value if it exists else load 0
            T val{static_cast<T>(0)};
            if (matrix_one_row < mat_one_rows && matrix_one_column < shared)
                val = matrix_one[matrix_one_row * stride_one + matrix_one_column];

#ifndef BENCHMARK
            static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % THREADS_PER_BLOCK == 0);
#endif
            // no bank conflict sequential access
            one_shared[shared_block_one_row][shared_block_one_column] = val;
        }

        const uint shared_two_iters{ceil_div(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)};

#pragma unroll
        for (uint load_idx{0}; load_idx < shared_two_iters; ++load_idx) {
            // the row of the matrix two shared block
            const size_t shared_block_two_row{
                (linear_idx + load_idx * THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_X
            };

            // the column of the matrix two shared block
            const size_t shared_block_two_column{
                (linear_idx + load_idx * THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_X
            };

            // the corresponding row in matrix two (this gets shifted by iteration and shared row value)
            const size_t matrix_two_row{iteration * BLOCK_TILE_SIZE_K + shared_block_two_row};

            // the corresponding column in matrix two this stays largely consistent
            const size_t matrix_two_column{blockIdx.x * BLOCK_TILE_SIZE_X + shared_block_two_column};

            T val{static_cast<T>(0)};
            if (matrix_two_row < shared && matrix_two_column < mat_two_columns)
                val = matrix_two[matrix_two_row * stride_two + matrix_two_column];

#ifndef BENCHMARK
            static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % THREADS_PER_BLOCK == 0);
#endif

            // no bank conflict sequential access
            two_shared[shared_block_two_row][shared_block_two_column] = val;
        }
    }

    template<typename T, uint BLOCK_TILE_SIZE_X, uint BLOCK_TILE_SIZE_Y, uint BLOCK_TILE_SIZE_K>
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
        /**
         * We start by creating two shared memory blocks, these memory blocks will act as a cache holding various
         * segments of the two matrices. Since vectors in a matrix are repeatedly used for computations of various
         * elements in the resulting matrix, we cache them in shared memory for easy computation.
         */
        __shared__ T mat_one_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        /**
         * All threads work in unison to load data, but are responsible for computing one element in the
         * dest matrix. So we compute the row and column of dest that this thread is responsible for.
         */
        const uint dest_column{threadIdx.x + blockDim.x * blockIdx.x};
        const uint dest_row{threadIdx.y + blockDim.y * blockIdx.y};

        T dummy{0};

        // ensure data is written within a boundary with minimal divergence overhead
        T *dest = (dest_row < mat_one_rows && dest_column < mat_two_columns)
                      ? &matrix_dest[row_stride_dest * dest_row + dest_column]
                      : &dummy;

        constexpr uint threads_per_block{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
        const uint linear_idx{threadIdx.y * BLOCK_TILE_SIZE_X + threadIdx.x};

        /**
         * the cache only holds segments of various rows and columns to do the full computation we are
         * new to cover the entire row. So we use a sliding window technique we slide matrix one to the right
         * across columns and we slide matrix two down by rows. total_iters calculates how many slides we need
         * to do
         */
        const uint total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        // the partial value after every slide
        T running_sum{static_cast<T>(0)};

        for (uint iter{0}; iter < total_iters; ++iter) {
            // load into shared memory

            // this function will handle loading data from global memory in the matrices to the respective
            // shared memory cache
            load_data_to_shared_memory<
                T,
                BLOCK_TILE_SIZE_X,
                BLOCK_TILE_SIZE_Y,
                BLOCK_TILE_SIZE_K,
                threads_per_block>(
                matrix_one,
                matrix_two,
                row_stride_one,
                row_stride_two,
                mat_one_thread_block_tile,
                mat_two_thread_block_tile,
                mat_one_rows,
                mat_two_columns,
                shared,
                iter,
                linear_idx);

            // ensure all threads are done loading to shared memory
            __syncthreads();

            // calculate intermediate value, we can do unrolling here due to less complexity and less registers needed
#pragma unroll
            for (size_t k{0}; k < BLOCK_TILE_SIZE_K; ++k) {
                /**
                 * comments on the memoru access parttern
                 * because we structured our blocks and grids to be (columns, row) this resulted in
                 * threads of the same warp sharing the same row (assuming rows are divisble by 32)
                 * this means that all threasd in the warp would do a broadcast for
                 * mat_one_thread_block_tile[threadIdx.y][k] since its the same spot for each thread, and
                 * mat_two_thread_block_tile[k][threadIdx.x] would result in no bank conflicts since positions would
                 * be adjacent.
                 */
                running_sum += mat_one_thread_block_tile[threadIdx.y][k] * mat_two_thread_block_tile[k][threadIdx.x];
            }

            // ensure all threads are done computing
            __syncthreads();
        }

        *dest *= beta;
        *dest += alpha * running_sum;

        // // after all iterations are complete save the value into the destination matrix
        // if (dest_column < mat_two_columns && dest_row < mat_one_rows) {
        //     matrix_dest[dest_row * row_stride_dest + dest_column] =
        //         (running_sum * alpha) + (beta * matrix_dest[dest_row * row_stride_dest + dest_column]);
        // }
    }

    // Used implementation by https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
    // double buffering cuda
    // undestand cache management

    template<typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_Y>
    __global__ void gemm_2DBT_1DTT(
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
        /**
         * same shared memory size as last time only now threads map to more elements in the block this is done by a
         * factor of THREAD_TILE_SIZE_Y. Essentially this means each thread computes THREAD_TILE_SIZE_Y elements across
         * the Y dimension.
         *
         * Heres a detailed breakdown of the thread mapping we want a block of size BLOCK_TILE_SIZE_Y x BLOCK_TILE_SIZE_X
         * Assume BLOCK_TILE_SIZE_Y = 8 and BLOCK_TILE_SIZE_X == 4 and tile size = 4, total elements of the block would be
         * 32 (8 * 4). Since tile size == 4 we divide total threads by that this leaves us with 8 total threads, the first
         * 4 threads will handle rows [0, 3] the second 4 threads will then handle row [4, 7] bringing us back to our
         * 8 rows.
         */
        __shared__ T mat_one_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        constexpr uint threads_per_block{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
        const uint thread_linear_idx{threadIdx.x}; // since block dim is 1 dimensional

        const uint total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        T intermediates[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

        // uint dest_col{threadIdx.x + BLOCK_TILE_SIZE_X * blockIdx.x};
        // uint dest_row(blockIdx.y * THREAD_TILE_SIZE_Y);

        for (uint iter{0}; iter < total_iters; ++iter) {
            load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, threads_per_block>(
                matrix_one,
                matrix_two,
                row_stride_one,
                row_stride_two,
                mat_one_thread_block_tile,
                mat_two_thread_block_tile,
                mat_one_rows,
                mat_two_columns,
                shared,
                iter,
                thread_linear_idx
            );
            __syncthreads();

#pragma unroll
            for (size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
                // save temporary value in a register and reuse it across thread tile
                // remember mat_two shared is of size BLOCK_TILE_SIZE_K x BLOCK_TILE_SIZE_X
                // so we are loading the elements in the representative column. This will be reused
                // THREAD_TILE_SIZE_Y times by each thread reducing memory access by a factor of THREAD_TILE_SIZE_Y
                // for matrix two
                // for example if we are computing position 0, 0 in the dest and tile size is 4 then we would also
                // compute 0, 1; 0,2; 0, 3

                // sequential access by threads in warp no bank conflicts if BLOCK_TILE_SIZE_X is a multiple of 32
                T two_val{mat_two_thread_block_tile[k_i][thread_linear_idx % BLOCK_TILE_SIZE_X]};
#pragma unroll
                for (size_t ij{0}; ij < THREAD_TILE_SIZE_Y; ++ij) {
                    // broadcast all threads in the same BLOCK_TILE_SIZE_X block use the same row values, ensure
                    // BLOCK_TILE_SIZE_X is at least 32 otherwise bank conflicts will appear since multiple rows will
                    // be accessed in the same warp
                    size_t row{thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y + ij};
                    intermediates[ij] += mat_one_thread_block_tile[row][k_i] * two_val;
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (size_t tile_row{0}; tile_row < THREAD_TILE_SIZE_Y; ++tile_row) {
            // commpute the sequential rows
            const size_t dest_row{
                blockIdx.y * BLOCK_TILE_SIZE_Y + thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y + tile_row
            };

            // column stays standard
            const size_t dest_col{thread_linear_idx % BLOCK_TILE_SIZE_X + BLOCK_TILE_SIZE_X * blockIdx.x};

            if (dest_row < mat_one_rows && dest_col < mat_two_columns) {
                matrix_dest[dest_row * row_stride_dest + dest_col] =
                        matrix_dest[dest_row * row_stride_dest + dest_col] * beta + intermediates[tile_row] * alpha;
            }
        }
    }

    template<typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X>
    __global__ void gemm_2DBT_2DTT(
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
        /**
         * in the previous iteration the only values cached were values from the mat two block, that too
         * only one singular value, in this case we will cache values from both matrices thus each thread
         * will be responsible for computing a matrix of elements, and will be responsible for loading a
         * matrix as well.
         */

        // each thread computes a matrix of element so we divide by the same amount
        constexpr uint threads_per_block{
            BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
        };

        __shared__ T mat_one_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        const uint total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        // the intermediate results of each computation
        T intermediates[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        // register cached values of matrix one
        T one_cache[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

        // register cached values of matrix two
        T two_cache[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        const uint thread_linear_idx{threadIdx.x}; // since block dim is 1 dimensional

        /**
         * High level overview of how this works
         *
         * Assume Matrix One
         * [0,  1,  2,  3]
         * [4,  5,  6,  7]
         * [8,  9,  10, 11]
         * [12, 13, 14, 15]
         *
         * Assume Matrix Two
         * [16, 17, 18, 19]
         * [20, 21, 22, 23]
         * [24, 25, 26, 27]
         * [28, 29, 30, 31]
         *
         * assume block
         * [0, 1]
         * [2, 3]
         *
         * thread 0 would load rowise from Matrix One in the first K step (into registers)
         * [0]
         * [4]
         * thread 0 would load column wise from Matrix two in the first K step (into registers)
         * [16, 17]
         *
         * it would then have compute the partial matrix
         * [0,  0]
         * [64, 68]
         *
         * It would then iterate over the respective K dimensions getting the final partial
         * matrices
         */

        for (uint iter{0}; iter < total_iters; ++iter) {
            load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, threads_per_block>(
                matrix_one,
                matrix_two,
                row_stride_one,
                row_stride_two,
                mat_one_thread_block_tile,
                mat_two_thread_block_tile,
                mat_one_rows,
                mat_two_columns,
                shared,
                iter,
                thread_linear_idx
            );
            __syncthreads();

            // add unrolling if less gpu pressure on 4070 my gpu theres too much register pressure for it be worth it
            // #pragma unroll
            for (size_t k_i{0}; k_i < BLOCK_TILE_SIZE_K; ++k_i) {
                // the starting row of mat one, used to load various elements
                const size_t mat_one_row{
                    thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y
                };

                // #pragma unroll
                for (size_t i{0}; i < THREAD_TILE_SIZE_Y; ++i) {
                    // because we divide the row by THREAD_TILE_SIZE_X the amount of threads per warp that represent
                    // multiple roes goes up, this leads to bank conflicts, to fix this one could load the mat_one
                    // shared block in a transposed manner.
                    one_cache[i] = mat_one_thread_block_tile[mat_one_row + i][k_i];
                }

                const size_t mat_two_col{
                    thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X
                };

                // #pragma unroll
                for (size_t j{0}; j < THREAD_TILE_SIZE_X; ++j) {
                    // no bank conflicts since all threads regardless of which row in the A matrix
                    // will start at the same row in the b matrix because of k_i
                    two_cache[j] = mat_two_thread_block_tile[k_i][mat_two_col + j];
                }

                for (size_t ki{0}; ki < THREAD_TILE_SIZE_Y; ++ki)
                    for (size_t kj{0}; kj < THREAD_TILE_SIZE_X; ++kj)
                        intermediates[ki][kj] += one_cache[ki] * two_cache[kj];
            }

            __syncthreads();
        }

        for (size_t thread_tile_y_idx{0}; thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx) {
            const size_t dest_row{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y + thread_tile_y_idx
            };

            for (size_t thread_tile_x_idx{0}; thread_tile_x_idx < THREAD_TILE_SIZE_X; ++thread_tile_x_idx) {
                const size_t dest_col{
                    BLOCK_TILE_SIZE_X * blockIdx.x +
                    thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_X + thread_tile_x_idx
                };

                if (dest_row < mat_one_rows && dest_col < mat_two_columns) {
                    matrix_dest[dest_row * row_stride_dest + dest_col] =
                            alpha * intermediates[thread_tile_y_idx][thread_tile_x_idx] +
                            beta * matrix_dest[dest_row * row_stride_dest + dest_col];
                }
            }
        }
    }

    /**
     * Loads data into the Matrix One and Two shared blocks
     * Matrix One will be stored transposed this is appropriate
     * if the algorithm originally access data columnwise leading
     * to bank conflicts. This is done with vectorized loads
     * improving memory bandwidth usage
     *
     * @tparam T the base datatype of the matrices
     * @tparam VECTOR_TYPE dictates how much data to load at once (defaults to 16 bytes)
     * @tparam BLOCK_TILE_SIZE_X
     * @tparam BLOCK_TILE_SIZE_Y
     * @tparam BLOCK_TILE_SIZE_K
     * @tparam THREADS_PER_BLOCK
     * @tparam BLOCK_TILE_SKEW_SIZE_X skews the starting positions for warp threads leading to less bank conflicts
     * @tparam BLOCK_TILE_SKEW_SIZE_Y skews the starting positions for warp threads leading to less bank conflicts
     * @param matrix_one
     * @param matrix_two
     * @param stride_one
     * @param stride_two
     * @param one_shared the shared memory block for matrix one should be transposed
     * @param two_shared the shared memory block for matrix two
     * @param mat_one_rows
     * @param mat_two_columns
     * @param shared
     * @param iteration
     * @param thread_linear_idx
     * @param v0 an initial zero value for vectorized loads defaulted too when bounds are exceeded
     */
    template<
        typename T,
        typename VECTOR_TYPE = int4,
        size_t BLOCK_TILE_SIZE_X,
        size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K,
        size_t THREADS_PER_BLOCK,
        size_t BLOCK_TILE_SKEW_SIZE_X = 0,
        size_t BLOCK_TILE_SKEW_SIZE_Y = 0
    >
    __device__ void load_data_to_shared_memory_transposed_vectorized(
        const T *matrix_one,
        const T *matrix_two,
        const size_t stride_one,
        const size_t stride_two,
        T one_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
        T two_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
        const size_t mat_one_rows,
        const size_t mat_two_columns,
        const size_t shared,
        const uint iteration,
        const uint thread_linear_idx,
        VECTOR_TYPE v0
    ) {
        constexpr size_t units_per_vector{sizeof(VECTOR_TYPE) / sizeof(T)};
        static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0);

        // ensure there will be an even amount of vectorized loads
        static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);
        static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);

#ifndef BENCHMARK
        // ensures leading dimensions are padded to handle additional reads
        assert(stride_one % units_per_vector == 0);
        assert(stride_two % units_per_vector == 0);
#endif

        // We need to make sure the data alignment is correct.
        static_assert((BLOCK_TILE_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
        static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

        static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
        static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

        // scaling the load number down to account for the vectorized size
        constexpr size_t VEC_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X / units_per_vector};
        constexpr size_t VEC_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / units_per_vector};

        // determines how many vectorized loads are performed per thread
        constexpr size_t one_iterations{
            ceil_div(BLOCK_TILE_SIZE_Y * VEC_BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)
        };

        // load into matrix one
#pragma unroll
        for (size_t one_iter{0}; one_iter < one_iterations; ++one_iter) {
            const size_t one_shared_row{(thread_linear_idx + one_iter * THREADS_PER_BLOCK) / VEC_BLOCK_TILE_SIZE_K};
            const size_t one_shared_column{
                (thread_linear_idx + one_iter * THREADS_PER_BLOCK) % VEC_BLOCK_TILE_SIZE_K * units_per_vector
            };

            const size_t mat_one_row{blockIdx.y * BLOCK_TILE_SIZE_Y + one_shared_row};
            const size_t mat_one_column{iteration * BLOCK_TILE_SIZE_K + one_shared_column};

            VECTOR_TYPE mat_one_row_vector_vals{v0};

            // if in bounds we save the data to the temp register value mat_one_row_vector_vals
            if (mat_one_row < mat_one_rows && mat_one_column < shared) {
                const VECTOR_TYPE *mat_one_vec_ptr{
                    reinterpret_cast<const VECTOR_TYPE *>(matrix_one + (mat_one_row * stride_one) + mat_one_column)
                };
                mat_one_row_vector_vals = *mat_one_vec_ptr;
            }

            // Transposed store of the data back into shared memory
            if (one_shared_row < BLOCK_TILE_SIZE_Y && one_shared_column < BLOCK_TILE_SIZE_K) {
                for (size_t i{0}; i < units_per_vector; ++i) {
                    one_shared[one_shared_column + i][one_shared_row] =
                            reinterpret_cast<const T *>(&mat_one_row_vector_vals)[i];
                }
            }
        }

        constexpr size_t two_iterations{ceil_div(BLOCK_TILE_SIZE_K * VEC_BLOCK_TILE_SIZE_X, THREADS_PER_BLOCK)};

        // load into matrix two
#pragma unroll
        for (size_t two_iter{0}; two_iter < two_iterations; ++two_iter) {
            const size_t two_shared_row{(thread_linear_idx + two_iter * THREADS_PER_BLOCK) / VEC_BLOCK_TILE_SIZE_X};

            const size_t two_shared_column{
                (thread_linear_idx + two_iter * THREADS_PER_BLOCK) % VEC_BLOCK_TILE_SIZE_X * units_per_vector
            };

            const size_t mat_two_row{iteration * BLOCK_TILE_SIZE_K + two_shared_row};
            const size_t mat_two_column{blockIdx.x * BLOCK_TILE_SIZE_X + two_shared_column};

            VECTOR_TYPE mat_two_row_vector_vals{v0};

            // if in bounds we save the data to the temp register value mat_two_row_vector_vals
            if (mat_two_row < shared && mat_two_column < mat_two_columns) {
                const VECTOR_TYPE *mat_two_vec_ptr{
                    reinterpret_cast<const VECTOR_TYPE *>(matrix_two + (mat_two_row * stride_two) + mat_two_column)
                };

                mat_two_row_vector_vals = *mat_two_vec_ptr;
            }

            if (two_shared_row < BLOCK_TILE_SIZE_K && two_shared_column < BLOCK_TILE_SIZE_X) {
                *reinterpret_cast<VECTOR_TYPE *>(&two_shared[two_shared_row][two_shared_column]) =
                        mat_two_row_vector_vals;
            }
        }
    }

    template<
        typename T,
        size_t BLOCK_TILE_SIZE_X,
        size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K,
        size_t THREAD_TILE_SIZE_X,
        size_t THREAD_TILE_SIZE_Y>
    __global__ void gemm_2DBT_2DTT_vload(
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
        constexpr size_t THREADS_PER_BLOCK{
            (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)
        };

        /**
         * TODO: possible improvements
         * Add the unrolling,
         * Make specific verisons for floats utilize float4 (can possibly lead to less instructions)
         *
         * Why not Double4
         * Most NVIDIA GPUs (especially consumer cards like RTX series) have much lower memory bandwidth and
         * ALU throughput for double precision, float 4 is the sweetspot
         */

        // using 1 dimensional block
        const size_t thread_linear_idx{threadIdx.x};

        // TRANSPOSED to avoid bank conflicts
        __shared__ T mat_one_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        const size_t total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        // the intermediate results of each computation
        T intermediates[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        // register cached values of matrix one
        T one_cache[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

        // register cached values of matrix two
        T two_cache[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        constexpr size_t units_per_vector{sizeof(int4) / sizeof(T)};

        // ensure int4 can be event split up by the base TYPE necessary for conversion
        static_assert(sizeof(int4) % sizeof(T) == 0);

        // we will store data along these dimensions for vectorized storage they need to be divisible
        static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);
        static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);
        constexpr size_t vectorized_thread_tile_size_x{THREAD_TILE_SIZE_X / units_per_vector};

        for (size_t iter{0}; iter < total_iters; ++iter) {
            // loads the data in a transposed manner for block A, this is necessary
            // since block A is transposed, this also loads data using vectorized loads
            // ain this case we use int4 which means essentially 16 bytes are loaded at once
            // this also requires 16 byte alignment
            load_data_to_shared_memory_transposed_vectorized<
                T, int4,
                BLOCK_TILE_SIZE_X,
                BLOCK_TILE_SIZE_Y,
                BLOCK_TILE_SIZE_K,
                THREADS_PER_BLOCK
            >(
                matrix_one,
                matrix_two,
                row_stride_one,
                row_stride_two,
                mat_one_thread_block_tile_transposed,
                mat_two_thread_block_tile,
                mat_one_rows,
                mat_two_columns,
                shared,
                iter,
                thread_linear_idx,
                int4{0, 0, 0, 0}
            );

            __syncthreads();

            // TODO test if unrolling is worth it? May add register pressure
            // # pragma unroll
            for (size_t k{0}; k < BLOCK_TILE_SIZE_K; ++k) {
                const size_t shared_block_row{
                    thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y
                };

                const size_t shared_block_column{
                    thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X
                };

                // we don't vectorize BLOCK A since the leading dimension is Y which may not be vectorization friendly
                // this is because we transposed the block to make it more friendly to bank conflict access
                // Imagine Tile Size of 2 and warp size of 2
                // Here is our Matrix
                // [10 20 30 40 ]
                // [30 40 50 60 ]
                // [50 60 70 80 ]
                // [70 80 90 100]
                //
                // Assume we had 4 threads
                // [0, 1, 2, 3]
                // Thread 0 and 1 will access 10 30 (columnwise); thread 2 and 3 will access 50 70 (columnwise)
                // this results in a bank conflict
                //
                // Transposing we get
                // [10 30 50 70]
                // [20 40 60 80]
                // [30 50 70 90]
                // [40 60 80 100]
                // Now accessing the values will result in a broadcast not a bank conflict

                // TODO test if unrolling is worth it? May add register pressure
                // #pragma unroll
                for (size_t one_tile_idx{0}; one_tile_idx < THREAD_TILE_SIZE_Y; ++one_tile_idx) {
                    one_cache[one_tile_idx] = mat_one_thread_block_tile_transposed[k][shared_block_row + one_tile_idx];
                }

                // here we do vectorized loads from shared memory directly into our registers for matrix B

                // TODO test if unrolling is worth it? May add register pressure
                // #pragma unroll
                for (size_t two_tile_idx{0}; two_tile_idx < vectorized_thread_tile_size_x; ++two_tile_idx) {
                    const auto b_shared_ptr{
                        reinterpret_cast<const int4 *>(&mat_two_thread_block_tile[k][shared_block_column]) +
                        two_tile_idx
                    };
                    reinterpret_cast<int4 *>(two_cache)[two_tile_idx] = *b_shared_ptr;
                }

                // We perform mat mul on create our partial result matrix
                for (size_t ki{0}; ki < THREAD_TILE_SIZE_Y; ++ki)
                    for (size_t kj{0}; kj < THREAD_TILE_SIZE_X; ++kj)
                        intermediates[ki][kj] += one_cache[ki] * two_cache[kj];
            }
            __syncthreads();
        }

        for (size_t y{0}; y < THREAD_TILE_SIZE_Y; ++y) {
            const size_t dest_row{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                (thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y) + y
            };

            for (size_t x{0}; x < vectorized_thread_tile_size_x; ++x) {
                const size_t dest_column{
                    blockIdx.x * BLOCK_TILE_SIZE_X +
                    (thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X) +
                    x * units_per_vector
                };

                auto dest_ptr{reinterpret_cast<int4 *>(&matrix_dest[dest_row * row_stride_dest + dest_column])};
                auto tile_ptr{reinterpret_cast<int4 *>(&intermediates[y][0]) + x};

                // we load the vectors from the tile and the Dest Matrix, we scale by alpha and beta
                // and perform the load at the end
                if (dest_row < mat_one_rows && dest_column < mat_two_columns) {
#pragma unroll
                    for (size_t i{0}; i < units_per_vector; ++i) {
                        reinterpret_cast<T *>(tile_ptr)[i] = reinterpret_cast<T *>(tile_ptr)[i] * alpha +
                                                             reinterpret_cast<T *>(dest_ptr)[i] * beta;
                    }
                    *dest_ptr = *tile_ptr;
                }
            }
        }
    }

    template<
        typename T,
        size_t BLOCK_TILE_SIZE_X,
        size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K,
        size_t THREAD_TILE_SIZE_X,
        size_t THREAD_TILE_SIZE_Y>
    __global__ void gemm_2DBT_2DTT_vload2(
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
        constexpr size_t THREADS_PER_BLOCK{
            (BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y) / (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)
        };
        /**
         * TODO: possible improvements
         * Add the unrolling,
         * Make specific verisons for floats utilize float4 (can possibly lead to less instructions)
         *
         * Why not Double4
         * Most NVIDIA GPUs (especially consumer cards like RTX series) have much lower memory bandwidth and
         * ALU throughput for double precision, float 4 is the sweetspot
         */

        // using 1 dimensional block
        const size_t thread_linear_idx{threadIdx.x};

        // TRANSPOSED to avoid bank conflicts
        __shared__ T mat_one_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        const size_t total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        // the intermediate results of each computation
        T intermediates[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        // register cached values of matrix one
        T one_cache[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

        // register cached values of matrix two
        T two_cache[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        constexpr size_t units_per_vector{sizeof(int4) / sizeof(T)};

        // ensure int4 can be event split up by the base TYPE necessary for conversion
        static_assert(sizeof(int4) % sizeof(T) == 0);

        // we will store data along these dimensions for vectorized storage they need to be divisible
        static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);
        static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);

        static_assert(THREAD_TILE_SIZE_X % units_per_vector == 0);
        static_assert(THREAD_TILE_SIZE_Y % units_per_vector == 0);

        constexpr size_t vectorized_thread_tile_size_x{THREAD_TILE_SIZE_X / units_per_vector};
        constexpr size_t vectorized_thread_tile_size_y{THREAD_TILE_SIZE_Y / units_per_vector};

        for (size_t iter{0}; iter < total_iters; ++iter) {
            // loads the data in a transposed manner for block A, this is necessary
            // since block A is transposed, this also loads data using vectorized loads
            // ain this case we use int4 which means essentially 16 bytes are loaded at once
            // this also requires 16 byte alignment
            load_data_to_shared_memory_transposed_vectorized<
                T, int4,
                BLOCK_TILE_SIZE_X,
                BLOCK_TILE_SIZE_Y,
                BLOCK_TILE_SIZE_K,
                THREADS_PER_BLOCK
            >(
                matrix_one,
                matrix_two,
                row_stride_one,
                row_stride_two,
                mat_one_thread_block_tile_transposed,
                mat_two_thread_block_tile,
                mat_one_rows,
                mat_two_columns,
                shared,
                iter,
                thread_linear_idx,
                int4{0, 0, 0, 0}
            );

            __syncthreads();

            // TODO test if unrolling is worth it? May add register pressure
            // # pragma unroll
            for (size_t k{0}; k < BLOCK_TILE_SIZE_K; ++k) {
                const size_t shared_block_row{
                    thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y
                };

                const size_t shared_block_column{
                    thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X
                };

                // we don't vectorize BLOCK A since the leading dimension is Y which may not be vectorization friendly
                // this is because we transposed the block to make it more friendly to bank conflict access
                // Imagine Tile Size of 2 and warp size of 2
                // Here is our Matrix
                // [10 20 30 40 ]
                // [30 40 50 60 ]
                // [50 60 70 80 ]
                // [70 80 90 100]
                //
                // Assume we had 4 threads
                // [0, 1, 2, 3]
                // Thread 0 and 1 will access 10 30 (columnwise); thread 2 and 3 will access 50 70 (columnwise)
                // this results in a bank conflict
                //
                // Transposing we get
                // [10 30 50 70]
                // [20 40 60 80]
                // [30 50 70 90]
                // [40 60 80 100]
                // Now accessing the values will result in a broadcast not a bank conflict

                // TODO test if unrolling is worth it? May add register pressure
                // #pragma unroll
                for (size_t one_tile_idx{0}; one_tile_idx < vectorized_thread_tile_size_y; ++one_tile_idx) {
                    const auto a_shared_ptr{
                        reinterpret_cast<const int4 *>(&mat_one_thread_block_tile_transposed[k][shared_block_row]) +
                        one_tile_idx
                    };

                    reinterpret_cast<int4 *>(one_cache)[one_tile_idx] = *a_shared_ptr;
                }

                // here we do vectorized loads from shared memory directly into our registers for matrix B

                // TODO test if unrolling is worth it? May add register pressure
                // #pragma unroll
                for (size_t two_tile_idx{0}; two_tile_idx < vectorized_thread_tile_size_x; ++two_tile_idx) {
                    const auto b_shared_ptr{
                        reinterpret_cast<const int4 *>(&mat_two_thread_block_tile[k][shared_block_column]) +
                        two_tile_idx
                    };
                    reinterpret_cast<int4 *>(two_cache)[two_tile_idx] = *b_shared_ptr;
                }

                // We perform mat mul on create our partial result matrix
                for (size_t ki{0}; ki < THREAD_TILE_SIZE_Y; ++ki)
                    for (size_t kj{0}; kj < THREAD_TILE_SIZE_X; ++kj)
                        intermediates[ki][kj] += one_cache[ki] * two_cache[kj];
            }
            __syncthreads();
        }

        for (size_t y{0}; y < THREAD_TILE_SIZE_Y; ++y) {
            const size_t dest_row{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                (thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y) + y
            };

            for (size_t x{0}; x < vectorized_thread_tile_size_x; ++x) {
                const size_t dest_column{
                    blockIdx.x * BLOCK_TILE_SIZE_X +
                    (thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X) +
                    x * units_per_vector
                };

                auto dest_ptr{reinterpret_cast<int4 *>(&matrix_dest[dest_row * row_stride_dest + dest_column])};
                auto tile_ptr{reinterpret_cast<int4 *>(&intermediates[y][0]) + x};

                // we load the vectors from the tile and the Dest Matrix, we scale by alpha and beta
                // and perform the load at the end
                if (dest_row < mat_one_rows && dest_column < mat_two_columns) {
#pragma unroll
                    for (size_t i{0}; i < units_per_vector; ++i) {
                        reinterpret_cast<T *>(tile_ptr)[i] = reinterpret_cast<T *>(tile_ptr)[i] * alpha +
                                                             reinterpret_cast<T *>(dest_ptr)[i] * beta;
                    }
                    *dest_ptr = *tile_ptr;
                }
            }
        }
    }

    template<
        typename T,
        size_t BLOCK_TILE_SIZE_X,
        size_t BLOCK_TILE_SIZE_Y,
        size_t BLOCK_TILE_SIZE_K,
        size_t WARP_TILE_SIZE_X,
        size_t WARP_TILE_SIZE_Y,
        size_t THREAD_TILE_SIZE_X,
        size_t THREAD_TILE_SIZE_Y,
        size_t NUM_THREADS_PER_WARP_X,
        size_t NUM_THREADS_PER_WARP_Y
    >
    __global__ void gemm_2DBT_2DWT_2DTT_vload(
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
        __shared__ T mat_one_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
        __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

        // One Warp TILE will be of size WARP_TILE_SIZE_X x WARP_TILE_SIZE_Y
        // One Warp will be responsible for each Warp block, ideally multiple warp blocks
        // will be able to fit in one regular block allowing multiple warps to exist per
        // block

        // EACH block computes BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y outputs of C

        // max threads per warp is 32, so we ensure that the warp block also complies
        // with this.
        static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

        // We need to figure out a couple of things,
        // 1) we need to figure out how many WARP Tiles will be present in
        // the x and y direction similar as to calculating how many blocks will
        // be in the grid for a GPU launch we are doing the same but making a block
        // the grid and having our WARP TILE Be the new block
        //
        // 2) This is needed to calculate the total amount of THREADS per block in a
        // constant way
        constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
        static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);

        // repeat for y dimension
        constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
        static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

        // so total amount of warp tiles in a block would be
        // NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y

        // In the previous implementation each thread had 2 register caches
        //
        // one cache cached several values in the y dimension from matrix one, total
        // elements are THREAD_TILE_SIZE_Y
        //
        // two cache cached several values in the x dimension from matrix two, total
        // elements are THREAD_TILE_SIZE_X
        //
        // In the end these values were reused for multiplication computing a total
        // of THREAD_TILE_SIZE_Y x THREAD_TILE_SIZE_X elements.
        //
        // Now that we are bounding warps to WARP_TILE sizes the amount of data being
        // computed by each thread may go up. So we know need to adjust the amount
        // of values being cached and computed to reflect this.
        //
        // We ideally want to keep our thread tile sizes consistent so we
        // instead add an extra dimension to each cache
        constexpr size_t NUM_CACHES_PER_WARP_X{
            WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)
        };

        // repeat for y TILE cache
        constexpr size_t NUM_CACHES_PER_WARP_Y{
            WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)
        };

        static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
        static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

        // Now we create the caches with the extra dimension
        T one_cache[NUM_CACHES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
        T two_cache[NUM_CACHES_PER_WARP_X][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        // since we have more caches we will have more intermediates (values computed per thread)
        // as well, so we add extra dimensions here as well reflecting this
        T intermediates[NUM_CACHES_PER_WARP_Y][NUM_CACHES_PER_WARP_X][THREAD_TILE_SIZE_Y][
            THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        // now we can also easily calculate the total threads per block, needed for loading data
        constexpr size_t THREADS_PER_BLOCK{NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y * 32};

        // this kernel should be launched with a 1d block so the linear dimension is just the threadidx.x
        const size_t thread_linear_idx{threadIdx.x};

        // the linear idx of the warp in the thread block
        const size_t warp_linear_idx{thread_linear_idx / warpSize};

        // Now lets figure out what warp that linear idx maps too (x, y)
        const size_t warp_row_idx{warp_linear_idx / NUM_WARPS_PER_BLOCK_X};
        const size_t warp_col_idx{warp_linear_idx % NUM_WARPS_PER_BLOCK_X};

        // figure out what row and column we are in the warp
        const size_t thread_linear_idx_in_warp{thread_linear_idx % warpSize};
        const size_t thread_idx_in_warp_row{thread_linear_idx_in_warp / NUM_THREADS_PER_WARP_X};
        const size_t thread_idx_in_warp_column{thread_linear_idx_in_warp % NUM_THREADS_PER_WARP_X};

        constexpr size_t units_per_vector{sizeof(int4) / sizeof(T)};

        // ensure int4 can be event split up by the base TYPE necessary for conversion
        static_assert(sizeof(int4) % sizeof(T) == 0);

        // we will store data along these dimensions for vectorized storage they need to be divisible
        static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);
        static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);

        static_assert(THREAD_TILE_SIZE_X % units_per_vector == 0);
        static_assert(THREAD_TILE_SIZE_Y % units_per_vector == 0);

        // This determines how many vectorized loads we need to perform to fill one tile
        constexpr size_t vectorized_thread_tile_size_x{THREAD_TILE_SIZE_X / units_per_vector};
        constexpr size_t vectorized_thread_tile_size_y{THREAD_TILE_SIZE_Y / units_per_vector};

        const size_t total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

        for (size_t iter{0}; iter < total_iters; ++iter) {
            load_data_to_shared_memory_transposed_vectorized<
                T, int4,
                BLOCK_TILE_SIZE_X,
                BLOCK_TILE_SIZE_Y,
                BLOCK_TILE_SIZE_K,
                THREADS_PER_BLOCK
            >(
                matrix_one,
                matrix_two,
                row_stride_one,
                row_stride_two,
                mat_one_thread_block_tile_transposed,
                mat_two_thread_block_tile,
                mat_one_rows,
                mat_two_columns,
                shared,
                iter,
                thread_linear_idx,
                int4{0, 0, 0, 0}
            );

            __syncthreads();

            // #pragma unroll
            for (size_t k{0}; k < BLOCK_TILE_SIZE_K; ++k) {
                // we need to start filling the one matrix cache
#pragma unroll
                for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
                    // Here we calculate the row in the shared block based on the warp coordinates
                    // and the thread coordinates

                    // To calculate the row we first multiply the warp block y coordinate by the
                    // Warp y dimension scale on the grid scale this is equivalent to doing blockIdx.y * blockDim.y
                    // Next based on what cache we are in we need to skip that many rows. We do this by multiplying the
                    // y_cache_idx by (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) this value is equivalent too
                    // (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) which is how many distinct rows are processed per
                    // one cache fill of warp tile. Finally, we add the row that this thread is part of in the warp.

                    // each thread loads TILE_SIZE_Y
                    // assuming this configuration NUM_THREADS_PER_WARP_X = 4, and NUM_THREADS_PER_WARP_Y = 8
                    // we can assume this load pattern
                    // Threads [0 to 3] load rows [0 to 7], Threads [4 to 7] load rows [8 to 15] ...
                    // Threads [28 to 31] load rows [54 to 63], this would result in a bank conflict for each
                    // new warp_row and a broadcast for all threads in warp row, but luckily
                    // the shared memory is transposed resulting in only broadcasts
                    const size_t one_shared_row_idx{
                        warp_row_idx * WARP_TILE_SIZE_Y +
                        y_cache_idx * (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) +
                        thread_idx_in_warp_row * THREAD_TILE_SIZE_Y
                    };

                    const auto one_shared_ptr{
                        reinterpret_cast<int4 *>(&mat_one_thread_block_tile_transposed[k][one_shared_row_idx])
                    };

                    auto tile_ptr{
                        reinterpret_cast<int4 *>(&one_cache[y_cache_idx])
                    };

                    // load into register cache one[y_cache_idx] with vectorized loads
                    // #pragma unroll
                    for (size_t vy_iter{0}; vy_iter < vectorized_thread_tile_size_y; ++vy_iter)
                        tile_ptr[vy_iter] = one_shared_ptr[vy_iter];
                }

#pragma unroll
                for (size_t x_cache_id{0}; x_cache_id < NUM_CACHES_PER_WARP_X; ++x_cache_id) {
                    const size_t two_shared_col_idx{
                        warp_col_idx * WARP_TILE_SIZE_X +
                        x_cache_id * (WARP_TILE_SIZE_X / NUM_CACHES_PER_WARP_X) +
                        thread_idx_in_warp_column * THREAD_TILE_SIZE_X
                    };

                    const auto two_shared_ptr{
                        reinterpret_cast<int4 *>(&mat_two_thread_block_tile[k][two_shared_col_idx])
                    };

                    auto tile_ptr{
                        reinterpret_cast<int4 *>(&two_cache[x_cache_id])
                    };

                    // #pragma unroll
                    for (size_t vx_iter{0}; vx_iter < vectorized_thread_tile_size_x; ++vx_iter)
                        tile_ptr[vx_iter] = two_shared_ptr[vx_iter];
                }

                // compute intermediates
#pragma unroll
                for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
                    // #pragma unroll
                    for (size_t x_cache_idx{0}; x_cache_idx < NUM_CACHES_PER_WARP_X; ++x_cache_idx) {
#pragma unroll
                        for (size_t one_cache_idx{0}; one_cache_idx < THREAD_TILE_SIZE_Y; ++one_cache_idx) {
                            T one_cache_value{one_cache[y_cache_idx][one_cache_idx]};
                            // #pragma unroll
                            for (size_t two_cache_index{0}; two_cache_index < THREAD_TILE_SIZE_X; ++two_cache_index) {
                                intermediates[y_cache_idx][x_cache_idx][one_cache_idx][two_cache_index] +=
                                        one_cache_value * two_cache[x_cache_idx][two_cache_index];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        // vectorized store back into the dest matrix
#pragma unroll
        for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
#pragma unroll
            for (size_t x_cache_idx{0}; x_cache_idx < NUM_CACHES_PER_WARP_X; ++x_cache_idx) {
                // #pragma unroll
                for (size_t one_cache_idx{0}; one_cache_idx < THREAD_TILE_SIZE_Y; ++one_cache_idx) {
                    const size_t dest_row{
                        BLOCK_TILE_SIZE_Y * blockIdx.y +
                        warp_row_idx * WARP_TILE_SIZE_Y +
                        y_cache_idx * (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) +
                        thread_idx_in_warp_row * THREAD_TILE_SIZE_Y + one_cache_idx
                    };

                    const size_t dest_column{
                        BLOCK_TILE_SIZE_X * blockIdx.x +
                        warp_col_idx * WARP_TILE_SIZE_X +
                        x_cache_idx * (WARP_TILE_SIZE_X / NUM_CACHES_PER_WARP_X) +
                        thread_idx_in_warp_column * THREAD_TILE_SIZE_X
                    };

                    auto dest_ptr{&matrix_dest[dest_row * row_stride_dest + dest_column]};
                    T *tile_ptr{&intermediates[y_cache_idx][x_cache_idx][one_cache_idx][0]};

                    // #pragma unroll
                    for (size_t two_cache_vec_idx{0}; two_cache_vec_idx < vectorized_thread_tile_size_x; ++
                         two_cache_vec_idx) {
                        if (dest_row < mat_one_rows && (
                                dest_column + two_cache_vec_idx * units_per_vector < mat_two_columns)) {
                            // #pragma unroll
                            for (size_t tile_idx{0}; tile_idx < units_per_vector; ++tile_idx) {
                                tile_ptr[tile_idx] = tile_ptr[tile_idx] * alpha + dest_ptr[tile_idx] * beta;
                            }

                            reinterpret_cast<int4 *>(dest_ptr)[two_cache_vec_idx] =
                                    reinterpret_cast<int4 *>(tile_ptr)[two_cache_vec_idx];
                        }
                    }
                }
            }
        }
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
                constexpr uint BLOCK_TILE_SIZE_X{16};
                constexpr uint BLOCK_TILE_SIZE_Y{16};
                constexpr uint BLOCK_TILE_SIZE_K{16};

                constexpr uint TOTAL_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};

                /**
                 * each thread will be responsible for loading elements into the two shared matrices
                 * for this to be possible the amount of threads present per block needs to be divisible
                 * by the shape of the two shared matrices. This is a parameter that can be scaled up and down
                 * for example if TOATL THREADS is equal to the shared matrix sizes each thread will be responsible
                 * for loading one element in each matrix. This is the implementation for block tiling. If we scale
                 * thread count down, this leads to more threads doign more wokr proportianately, both loading
                 * compute.
                 */

                static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % TOTAL_THREADS == 0);
                static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % TOTAL_THREADS == 0);

                constexpr dim3 block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1};

                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };

                gemm_tiled_2<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K><<<grid_dim, block_dim>>>(
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
            case 3: {
                /**
                 * in the previous implementation to compute one element we would need to do read k elements from One and
                 * k elements from Two. With this implementation to compute one tile, we need to read k x tile for A but only
                 * k for b, so to compute one element on average we need k + k / tile, which is less than 2k for the orignal.
                 * So we reduce shared memeory pressure by alot.
                 */
                constexpr uint BLOCK_TILE_SIZE_X{64};
                constexpr uint BLOCK_TILE_SIZE_Y{64};
                constexpr uint BLOCK_TILE_SIZE_K{8};

                constexpr uint THREAD_TILE_SIZE_Y{8};

                // scale down threads by a factor of THREAD_TILE_SIZE_Y since each thread will be processing that
                // many values
                constexpr uint NUM_THREADS_PER_BLOCK{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};

                // needs to be divisible otherwise threads will compute an uneven amount of data
                static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

                // needs to be true for thread mapping in matrix one block
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0);

                // needs to be true to determine thread mapping for the b block and dest matrix
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0);

                constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);
                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };

                gemm_2DBT_1DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y><<<
                        grid_dim, block_dim>>>(
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
            case 4: {
                constexpr uint BLOCK_TILE_SIZE_X{128};
                constexpr uint BLOCK_TILE_SIZE_Y{128};
                constexpr uint BLOCK_TILE_SIZE_K{16};

                constexpr uint THREAD_TILE_SIZE_Y{8};
                constexpr uint THREAD_TILE_SIZE_X{8};

                constexpr uint NUM_THREADS_PER_BLOCK{
                    BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
                };

                static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
                static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0);
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0);

                static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0);
                static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0);

                constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);
                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };

                gemm_2DBT_2DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y,
                    THREAD_TILE_SIZE_X><<<grid_dim, block_dim>>>(
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
            case 5: {
                // mainly dictates how much shared memory we use
                constexpr uint BLOCK_TILE_SIZE_X{128};
                constexpr uint BLOCK_TILE_SIZE_Y{128};
                constexpr uint BLOCK_TILE_SIZE_K{16};

                // Each thread is responsible for computing a matrix block of c
                // the size of which is THREAD_TILE_SIZE_Y x THREAD_TILE_SIZE_X
                constexpr uint THREAD_TILE_SIZE_Y{8};
                constexpr uint THREAD_TILE_SIZE_X{8};

                // In total each block should compute BLOCK_TILE_SIZE_X x BLOCK_TILE_SIZE_Y
                // elements of C, since each thread is responsible for BLOCK_TILE_SIZE_X x BLOCK_TILE_SIZE_Y
                // we calculate the total threads based on that
                constexpr uint NUM_THREADS_PER_BLOCK{
                    BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
                };

                // we ensure that Tiles are evenly distributed among threads
                static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
                static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

                // TODO why?
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0);
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0);

                // This ensures that all threads are responsible for loading the same amount of data
                static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0);
                static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0);

                // 1 dimensional thread count
                constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };

                // vectorized loads only work when multiple loads are needed to satisfy a tile
                // for example the load size we use int4 (4 ints = 16 bytes) tile size is 8 = 32 bytes
                // two loads are required. Now lets say int8 which is 1 byte, tile size is 8 which is 8 bytes
                // the load with int4 will result in 16 bytes being loaded which is 16 elements this is spillover
                if constexpr (constexpr size_t byts{sizeof(T)}; byts < 2) {
                    gemm_2DBT_2DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X><<<grid_dim, block_dim>>>(
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
                } else {
                    gemm_2DBT_2DTT_vload<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X,
                        THREAD_TILE_SIZE_Y><<<grid_dim, block_dim>>>(
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
                }

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 6: {
                // mainly dictates how much shared memory we use
                constexpr uint BLOCK_TILE_SIZE_X{128};
                constexpr uint BLOCK_TILE_SIZE_Y{128};
                constexpr uint BLOCK_TILE_SIZE_K{16};

                // Each thread is responsible for computing a matrix block of c
                // the size of which is THREAD_TILE_SIZE_Y x THREAD_TILE_SIZE_X
                constexpr uint THREAD_TILE_SIZE_Y{8};
                constexpr uint THREAD_TILE_SIZE_X{8};

                // In total each block should compute BLOCK_TILE_SIZE_X x BLOCK_TILE_SIZE_Y
                // elements of C, since each thread is responsible for BLOCK_TILE_SIZE_X x BLOCK_TILE_SIZE_Y
                // we calculate the total threads based on that
                constexpr uint NUM_THREADS_PER_BLOCK{
                    BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
                };

                // we ensure that Tiles are evenly distributed among threads
                static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0);
                static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0);

                // TODO why?
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0);
                static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0);

                // This ensures that all threads are responsible for loading the same amount of data
                static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0);
                static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0);

                // 1 dimensional thread count
                constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };

                // vectorized loads only work when multiple loads are needed to satisfy a tile
                // for example the load size we use int4 (4 ints = 16 bytes) tile size is 8 = 32 bytes
                // two loads are required. Now lets say int8 which is 1 byte, tile size is 8 which is 8 bytes
                // the load with int4 will result in 16 bytes being loaded which is 16 elements this is spillover
                if constexpr (constexpr size_t byts{sizeof(T)}; byts < 2) {
                    gemm_2DBT_2DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X><<<grid_dim, block_dim>>>(
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
                } else {
                    gemm_2DBT_2DTT_vload2<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X
                        ,
                        THREAD_TILE_SIZE_Y><<<grid_dim, block_dim>>>(
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
                }

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 7: {
                constexpr uint BLOCK_TILE_SIZE_X{128};
                constexpr uint BLOCK_TILE_SIZE_Y{128};
                constexpr uint BLOCK_TILE_SIZE_K{16};

                // the size of the warp tile block, each warp is responsible for computing
                // this many elements
                constexpr unsigned int WARP_TILE_SIZE_X{64};
                constexpr unsigned int WARP_TILE_SIZE_Y{64};

                // check if warp tile blocks fit evenly in the regular block
                constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
                constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

                static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
                static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

                // The size of the internal register caches
                constexpr uint THREAD_TILE_SIZE_Y{8};
                constexpr uint THREAD_TILE_SIZE_X{8};

                // how many threads to allocate in each dimension, the product must
                // be 32
                constexpr unsigned int NUM_THREADS_PER_WARP_X{4};
                constexpr unsigned int NUM_THREADS_PER_WARP_Y{8};

                static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

                // ensure each thread stores the same amount of data in their tiles
                static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
                static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

                const dim3 grid_dim{
                    ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
                    ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
                };

                if constexpr (constexpr size_t byts{sizeof(T)}; byts < 2) {
                    constexpr uint NUM_THREADS_PER_BLOCK{
                        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
                    };

                    constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

                    gemm_2DBT_2DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X><<<grid_dim, block_dim>>>(
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
                } else {
                    constexpr size_t NUM_THREADS_PER_BLOCK{32 * NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y};
                    constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

                    gemm_2DBT_2DWT_2DTT_vload<
                        T,
                        BLOCK_TILE_SIZE_X,
                        BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K,
                        WARP_TILE_SIZE_X,
                        WARP_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X,
                        THREAD_TILE_SIZE_Y,
                        NUM_THREADS_PER_WARP_X,
                        NUM_THREADS_PER_WARP_Y><<<grid_dim, block_dim>>>(
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
                        row_stride_dest
                    );
                }

                CUDA_CHECK(cudaGetLastError());
                return;
            }
            case 8: {
                if constexpr (std::is_same_v<T, float>) {
                    gemm_cublas(
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
                return;
            }
            default: {
                throw std::runtime_error("invalid function provided");
            }
        }
#else
        constexpr uint BLOCK_TILE_SIZE_X{128};
        constexpr uint BLOCK_TILE_SIZE_Y{128};
        constexpr uint BLOCK_TILE_SIZE_K{16};

        // the size of the warp tile block, each warp is responsible for computing
        // this many elements
        constexpr unsigned int WARP_TILE_SIZE_X{64};
        constexpr unsigned int WARP_TILE_SIZE_Y{64};

        // check if warp tile blocks fit evenly in the regular block
        constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
        constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

        static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
        static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

        // The size of the internal register caches
        constexpr uint THREAD_TILE_SIZE_Y{8};
        constexpr uint THREAD_TILE_SIZE_X{8};

        // how many threads to allocate in each dimension, the product must
        // be 32
        constexpr unsigned int NUM_THREADS_PER_WARP_X{4};
        constexpr unsigned int NUM_THREADS_PER_WARP_Y{8};

        static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

        // ensure each thread stores the same amount of data in their tiles
        static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
        static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

        const dim3 grid_dim{
            ceil_div(static_cast<uint>(mat_two_columns), BLOCK_TILE_SIZE_X),
            ceil_div(static_cast<uint>(mat_one_rows), BLOCK_TILE_SIZE_Y)
        };

        if constexpr (constexpr size_t byts{sizeof(T)}; byts < 2) {
            constexpr uint NUM_THREADS_PER_BLOCK{
                BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)
            };

            constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

            gemm_2DBT_2DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y,
                THREAD_TILE_SIZE_X><<<grid_dim, block_dim>>>(
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
        } else {
            constexpr size_t NUM_THREADS_PER_BLOCK{32 * NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y};
            constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

            gemm_2DBT_2DWT_2DTT_vload<
                T,
                BLOCK_TILE_SIZE_X,
                BLOCK_TILE_SIZE_Y,
                BLOCK_TILE_SIZE_K,
                WARP_TILE_SIZE_X,
                WARP_TILE_SIZE_Y,
                THREAD_TILE_SIZE_X,
                THREAD_TILE_SIZE_Y,
                NUM_THREADS_PER_WARP_X,
                NUM_THREADS_PER_WARP_Y><<<grid_dim, block_dim>>>(
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
                row_stride_dest
            );
        }

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
