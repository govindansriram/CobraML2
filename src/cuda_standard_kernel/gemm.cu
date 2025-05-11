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

    __device__ __forceinline__ uint d_ceil_div(const uint a, const uint b) {
        return (a + b - 1) / b;
    }

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
        const uint shared_one_iters{d_ceil_div(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)};
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

        const uint shared_two_iters{d_ceil_div(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)};

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
        const uint total_iters{d_ceil_div(shared, BLOCK_TILE_SIZE_K)};

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

        const uint total_iters{d_ceil_div(shared, BLOCK_TILE_SIZE_K)};

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

        const uint total_iters{d_ceil_div(shared, BLOCK_TILE_SIZE_K)};

        // the intermediate results of each computation
        T intermediates[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        // register cached values of matrix one
        T one_cache[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

        // register cached values of matrix two
        T two_cache[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

        const uint thread_linear_idx{threadIdx.x}; // since block dim is 1 dimensional

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

                gemm_2DBT_1DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y><<<grid_dim, block_dim>>>(
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
                    BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X)};

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

                gemm_2DBT_2DTT<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_Y, THREAD_TILE_SIZE_X><<<grid_dim, block_dim>>>(
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
