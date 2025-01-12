//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include <cstring>
#include <iostream>
#include "../math_dis.h"


namespace cobraml::core {
    template<typename NumType>
    void gemv_naive(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        for (size_t start = 0; start < rows; ++start) {
            for (size_t i = 0; i < columns; ++i) {
                dest[start] = vector[i] * matrix[start * columns + i];
            }
        }
    }

    template<typename NumType>
    void gemv_parallel(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        size_t start;

#pragma omp parallel for default(none) shared(matrix, vector, dest, rows, columns) private(start) schedule(dynamic)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;
#pragma omp simd reduction(+:partial)
            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }

            dest[start] = partial;
        }
    }

    template<typename NumType>
    void gemv_parallel_block(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        constexpr size_t block_rows{8}; // best 15 // 8
        constexpr size_t block_columns{8192 / sizeof(NumType)};

        size_t blocks_per_row{columns / block_columns};
        blocks_per_row += columns % block_columns > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t blocks_per_column{rows / block_rows};
        blocks_per_column += rows % block_rows > 0 ? 1 : 0; // add one more block if there is a remainder

        auto *dest_partials = new NumType[rows * blocks_per_row]();

        for (size_t i = 0; i < blocks_per_row; ++i) {
            const NumType *vector_segment{&vector[i * block_columns]};
            size_t vector_len{block_columns};

            if (i == blocks_per_row - 1) {
                vector_len = columns - (block_columns * i);
            }

            size_t j;

#pragma omp parallel for default(none) shared(dest_partials, block_rows, blocks_per_column, blocks_per_row, vector_segment, vector_len, matrix, dest, columns, rows, i) private(j) schedule(dynamic)
            for (j = 0; j < blocks_per_column; ++j) {
                size_t row_start{j * block_rows};
                size_t row_end{row_start + block_rows};

                if (j == blocks_per_column - 1) {
                    row_end = row_start + rows - (block_rows * j);
                }

                for (; row_start < row_end; ++row_start) {
                    NumType partial{0};

                    for (size_t k = 0; k < vector_len; ++k) {
                        partial += static_cast<NumType>(
                            vector_segment[k] * matrix[row_start * columns + block_columns * i + k]);
                    }

                    dest_partials[row_start * blocks_per_row + i] += partial;
                }
            }
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < blocks_per_row; ++j) {
                dest[i] += dest_partials[i * blocks_per_row + j];
            }
        }

        delete[] dest_partials;
    }

#define BLOCK_COLUMNS_SIZE 8192
#define BLOCK_ROW_SIZE 4
#define CACHE_LINE_SIZE 64

    template<typename NumType>
    void gemv_parallel_block_2(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        constexpr size_t block_multiplier = 1;
        constexpr size_t block_columns = (10624 * block_multiplier) / sizeof(NumType);

        size_t blocks_per_row = columns / block_columns;
        blocks_per_row += columns % block_columns > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t i;
#pragma omp parallel for default(none) shared(dest, block_columns, blocks_per_row, vector, matrix, columns, rows) private(i) schedule(dynamic)
        for (i = 0; i < rows; ++i) {
            NumType partial{0};

            for (size_t j = 0; j < blocks_per_row; ++j) {
                const NumType *vector_segment = &vector[j * block_columns];
                size_t vector_len{block_columns};

                if (j == blocks_per_row - 1) {
                    vector_len = columns - (block_columns * j);
                }

// #pragma omp simd reduction(+:partial)
                for (size_t k = 0; k < vector_len; ++k) {
                    partial = static_cast<NumType>(partial +
                        vector_segment[k] * matrix[(i * columns) + (j * block_columns) + k]);
                }
            }

            dest[i] = partial;
        }
    }

    template<typename NumType>
    void gemv_parallel_block_copy(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const size_t rows,
        const size_t columns) {
        constexpr size_t columns_per_block{BLOCK_COLUMNS_SIZE / sizeof(NumType)};

        size_t blocks_per_row{columns / columns_per_block};
        blocks_per_row += columns % columns_per_block > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t row_blocks{rows / BLOCK_ROW_SIZE};
        row_blocks += rows % BLOCK_ROW_SIZE > 0 ? 1 : 0; // add one more block if there is a remainder

        size_t cache_lines_per_row_block = BLOCK_ROW_SIZE * sizeof(NumType) / CACHE_LINE_SIZE;
        cache_lines_per_row_block += BLOCK_ROW_SIZE * sizeof(NumType) % CACHE_LINE_SIZE > 0 ? 1 : 0;

        auto *dest_partials = static_cast<char *>(
            std::aligned_alloc(
                CACHE_LINE_SIZE,
                CACHE_LINE_SIZE * cache_lines_per_row_block * row_blocks * blocks_per_row)
        );

        for (size_t i = 0; i < blocks_per_row; ++i) {
            const NumType *vector_segment{&vector[i * columns_per_block]};
            size_t vector_len{columns_per_block};

            if (i == blocks_per_row - 1) {
                vector_len = columns - (columns_per_block * i);
            }

            alignas(CACHE_LINE_SIZE) NumType local_vector[columns_per_block]{};

            memcpy(local_vector, vector_segment, vector_len * sizeof(NumType));

            char *temp_addr = &dest_partials[(i * CACHE_LINE_SIZE * cache_lines_per_row_block * row_blocks)];

            size_t j;

            alignas(CACHE_LINE_SIZE) NumType local_matrix[BLOCK_ROW_SIZE][columns_per_block];

#pragma omp parallel for default(none) shared(std::cout, vector_len, columns_per_block, temp_addr, cache_lines_per_row_block, row_blocks, blocks_per_row, matrix, dest, rows, i, columns) private(j, local_matrix) firstprivate(local_vector)
            for (j = 0; j < row_blocks; ++j) {
                size_t row_start{j * BLOCK_ROW_SIZE};
                size_t row_end{row_start + BLOCK_ROW_SIZE};

                if (j == row_blocks - 1) {
                    row_end = row_start + rows - (BLOCK_ROW_SIZE * j);
                }

                for (size_t c_row = 0; c_row < (row_end - row_start); ++c_row) {
                    memcpy(
                        local_matrix[c_row],
                        &matrix[(row_start + c_row) * columns + columns_per_block * i],
                        vector_len * sizeof(NumType));
                }

                auto addr = reinterpret_cast<NumType *>(&temp_addr[(j * CACHE_LINE_SIZE * cache_lines_per_row_block)]);

                for (int s = 0; row_start < row_end; ++row_start, ++s) {
                    NumType partial{0};

#pragma omp simd reduction(+:partial) aligned(local_matrix: CACHE_LINE_SIZE) aligned(local_vector: CACHE_LINE_SIZE)
                    for (size_t k = 0; k < columns_per_block; ++k) {
                        partial += static_cast<NumType>(
                            local_vector[k] * local_matrix[s][k]);
                    }

                    addr[s] = partial;
                }
            }
        }

        for (size_t i = 0; i < blocks_per_row; ++i) {
            char *temp_addr = &dest_partials[(i * CACHE_LINE_SIZE * cache_lines_per_row_block * row_blocks)];

            for (size_t j = 0; j < row_blocks; ++j) {
                auto addr = reinterpret_cast<NumType *>(&temp_addr[(j * CACHE_LINE_SIZE * cache_lines_per_row_block)]);
                size_t row_start{j * BLOCK_ROW_SIZE};
                size_t row_end{row_start + BLOCK_ROW_SIZE};

                if (j == row_blocks - 1) {
                    row_end = row_start + rows - (BLOCK_ROW_SIZE * j);
                }

                for (int s = 0; row_start < row_end; ++row_start, ++s) {
                    dest[row_start] += addr[s];
                }
            }
        }

        std::free(dest_partials);
    }

#ifdef BENCHMARK

    template<typename NumType>
    void benchmarked_gemv(
        const NumType *mat,
        const NumType *vec,
        NumType *dest,
        size_t const rows,
        size_t const columns) {
        switch (func_pos) {
            case 0: {
                gemv_naive(mat, vec, dest, rows, columns);
                return;
            }
            case 1: {
                gemv_parallel(mat, vec, dest, rows, columns);
                return;
            }
            case 2: {
                gemv_parallel_block(mat, vec, dest, rows, columns);
                return;
            }
            // case 3: {
            //     gemv_parallel_block_copy(mat, vec, dest, rows, columns);
            //     return;
            // }
            case 3: {
                gemv_parallel_block_2(mat, vec, dest, rows, columns);
                return;
            }
            default: {
                throw std::runtime_error("invalid gemv type provided");
            }
        }
    }

#else
    template<typename NumType>
    void benchmarked_gemv(
        const NumType *mat,
        const NumType *vec,
        NumType *dest,
        size_t const rows,
        size_t const columns) {

        gemv_parallel_block_2(mat, vec, dest, rows, columns);
    }
#endif

    class StandardMath final : public Math {
        void batched_dot_product(const void *matrix, const void *vector, void *dest, size_t rows, size_t columns,
                                 Dtype dtype) override;
    };
}

#endif //STANDARD_MATH_H
