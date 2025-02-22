//
// Created by Sriram Govindan on 12/23/24.
//

#include "standard_math.h"
#include <omp.h>
#include "enums.h"
#include <cstring>
#include <iostream>
#include <immintrin.h>
#include <cassert>
#include <cmath>

namespace cobraml::core {
    static void set_num_threads() {
        static bool set = false;

        if (!set) {
#ifdef NUM_THREADS
            omp_set_num_threads(NUM_THREADS);
#else
            omp_set_num_threads(omp_get_max_threads());
#endif
        }
    }

    void StandardMath::gemv(
        const void *matrix,
        const void *vector,
        void *dest,
        const void *alpha,
        const void *beta,
        size_t const rows,
        size_t const columns,
        Dtype const dtype) {
        set_num_threads();
        switch (dtype) {
            case FLOAT64: {
                const auto casted_dest = static_cast<double *>(dest);
                const auto casted_mat = static_cast<const double *>(matrix);
                const auto casted_vec = static_cast<const double *>(vector);
                const auto casted_alpha = static_cast<const double *>(alpha);
                const auto casted_beta = static_cast<const double *>(beta);
                benchmarked_gemv<double>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns);
                return;
            }
            case FLOAT32: {
                const auto casted_dest = static_cast<float *>(dest);
                const auto casted_mat = static_cast<const float *>(matrix);
                const auto casted_vec = static_cast<const float *>(vector);
                const auto casted_alpha = static_cast<const float *>(alpha);
                const auto casted_beta = static_cast<const float *>(beta);
                benchmarked_gemv<float>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns);
                return;
            }
            case INT8: {
                const auto casted_dest = static_cast<int8_t *>(dest);
                const auto casted_mat = static_cast<const int8_t *>(matrix);
                const auto casted_vec = static_cast<const int8_t *>(vector);
                const auto casted_alpha = static_cast<const int8_t *>(alpha);
                const auto casted_beta = static_cast<const int8_t *>(beta);
                benchmarked_gemv<int8_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns);
                return;
            }
            case INT16: {
                const auto casted_dest = static_cast<int16_t *>(dest);
                const auto casted_mat = static_cast<const int16_t *>(matrix);
                const auto casted_vec = static_cast<const int16_t *>(vector);
                const auto casted_alpha = static_cast<const int16_t *>(alpha);
                const auto casted_beta = static_cast<const int16_t *>(beta);
                benchmarked_gemv<int16_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns);
                return;
            }
            case INT32: {
                const auto casted_dest = static_cast<int32_t *>(dest);
                const auto casted_mat = static_cast<const int32_t *>(matrix);
                const auto casted_vec = static_cast<const int32_t *>(vector);
                const auto casted_alpha = static_cast<const int32_t *>(alpha);
                const auto casted_beta = static_cast<const int32_t *>(beta);
                benchmarked_gemv<int32_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns);
                return;
            }
            case INT64: {
                const auto casted_dest = static_cast<int64_t *>(dest);
                const auto casted_mat = static_cast<const int64_t *>(matrix);
                const auto casted_vec = static_cast<const int64_t *>(vector);
                const auto casted_alpha = static_cast<const int64_t *>(alpha);
                const auto casted_beta = static_cast<const int64_t *>(beta);
                benchmarked_gemv<int64_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate gemv on invalid type");
            }
        }
    }

#ifdef AVX2
#define SKIP 2
#define UNROLLS 2

    template<>
    void gemv_manual<float>(
        const float *matrix,
        const float *vector,
        float *dest,
        float const alpha,
        float const beta,
        size_t const rows,
        size_t const columns) {
        size_t start;
        size_t const row_count = get_row_count(rows, SKIP);
        constexpr size_t skip = get_block_len<float>();


#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, row_count, columns, skip) private(start) schedule(dynamic)
        for (start = 0; start < row_count / SKIP; ++start) {
            float partial_1 = 0;
            float partial_2 = 0;

            const size_t row_start = start * SKIP;

            for (size_t i = 0; i < columns; i += skip) {
                __m256 const vector_block = _mm256_load_ps(&vector[i]);
                __m256 const mat_block_1 = _mm256_load_ps(&matrix[start * columns + i]);
                __m256 const mat_block_2 = _mm256_load_ps(&matrix[(start + 1) * columns + i]);

                __m256 const result1 = _mm256_mul_ps(vector_block, mat_block_1);
                __m256 const result2 = _mm256_mul_ps(vector_block, mat_block_2);

                alignas(256) float temp1[skip];
                alignas(256) float temp2[skip];

                _mm256_store_ps(temp1, result1);
                _mm256_store_ps(temp2, result2);

                partial_1 += temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp1[4] + temp1[5] + temp1[6] + temp1[7];
                // test with haddps
                partial_2 += temp2[0] + temp2[1] + temp2[2] + temp2[3] + temp2[4] + temp2[5] + temp2[6] + temp2[7];
                // test with haddps
            }

            dest[row_start] = dest[row_start] * beta + partial_1 * alpha;
            dest[row_start + 1] = dest[row_start + 1] * beta + partial_2 * alpha;
        }

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, row_count, columns) private(start) schedule(dynamic)
        for (start = row_count; start < rows; ++start) {
            float partial = 0;


#pragma omp simd reduction(+:partial) aligned(vector: 32) aligned(matrix: 32)
            for (size_t i = 0; i < columns; ++i) {
                partial += vector[i] * matrix[start * columns + i];
            }

            dest[start] = dest[start] * beta + partial * alpha;
        }
    }

    template<>
    void gemv_manual<double>(
        const double *matrix,
        const double *vector,
        double *dest,
        double const alpha,
        double const beta,
        size_t const rows,
        size_t const columns) {
        size_t start;
        size_t const row_count{get_row_count(rows, SKIP)};
        constexpr size_t skip{get_block_len<double>()};
        constexpr size_t jump{UNROLLS * skip};
        const size_t column_count{columns / jump};

#ifndef BENCHMARK
        assert(reinterpret_cast<uintptr_t>(matrix) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(vector) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(dest) % 32 == 0);

        std::cout << "col count " << column_count << std::endl;;
        std::cout << "columns " << columns << std::endl;
        std::cout << "row count " << row_count << std::endl;;
        std::cout << "rows " << rows << std::endl;
#endif

#pragma omp parallel for default(none) shared(std::cout, column_count, alpha, beta, matrix, vector, dest, row_count, columns, skip) private(start) schedule(dynamic)
        for (start = 0; start < row_count / SKIP; ++start) {
            double partial_1 = 0;
            double partial_2 = 0;

            const size_t row_start = start * SKIP;

            for (size_t i = 0; i < column_count; i += 1) {
                __m256d const vector_block1 = _mm256_load_pd(&vector[i * jump]);
                __m256d const vector_block2 = _mm256_load_pd(&vector[i * jump + skip]);

                __m256d const mat_block_1_1 = _mm256_loadu_pd(&matrix[row_start * columns + (i * jump)]);
                __m256d const mat_block_1_2 = _mm256_loadu_pd(&matrix[row_start * columns + (i * jump) + skip]);
                __m256d const mat_block_2_1 = _mm256_loadu_pd(&matrix[(row_start + 1) * columns + (i * jump)]);
                __m256d const mat_block_2_2 = _mm256_loadu_pd(&matrix[(row_start + 1) * columns + (i * jump) + skip]);

                __m256d result1_1 = _mm256_mul_pd(vector_block1, mat_block_1_1);
                __m256d result2_1 = _mm256_mul_pd(vector_block1, mat_block_2_1);
                __m256d const result1_2 = _mm256_mul_pd(vector_block2, mat_block_1_2);
                __m256d const result2_2 = _mm256_mul_pd(vector_block2, mat_block_2_2);

                result1_1 = _mm256_add_pd(result1_1, result1_2);
                result2_1 = _mm256_add_pd(result2_1, result2_2);

                alignas(32) double temp1[skip];
                alignas(32) double temp2[skip];

                _mm256_store_pd(temp1, result1_1);
                _mm256_store_pd(temp2, result2_1);

                partial_1 += temp1[0] + temp1[1] + temp1[2] + temp1[3];
                partial_2 += temp2[0] + temp2[1] + temp2[2] + temp2[3];

                // std::cout << "here: " << i << std::endl;
            }

            // cleanup remainders
            for (size_t i = column_count * jump; i < columns; i += skip) {
                __m256d mat_block_1;
                __m256d mat_block_2;
                __m256d const vector_block = _mm256_load_pd(&vector[i]);

                // std::cout << i << "\n";

                if (i + skip > columns) {
                    size_t const amt = columns - i;
                    alignas(32) double block_1[skip]{};
                    alignas(32) double block_2[skip]{};

                    std::memcpy(block_1, &matrix[row_start * columns + i], amt * 8);
                    std::memcpy(block_2, &matrix[(row_start + 1) * columns + i], amt * 8);

                    mat_block_1 = _mm256_load_pd(block_1);
                    mat_block_2 = _mm256_load_pd(block_2);
                } else {
                    mat_block_1 = _mm256_loadu_pd(&matrix[row_start * columns + i]);
                    mat_block_2 = _mm256_loadu_pd(&matrix[(row_start + 1) * columns + i]);
                }

                __m256d const result1 = _mm256_mul_pd(vector_block, mat_block_1);
                __m256d const result2 = _mm256_mul_pd(vector_block, mat_block_2);

                alignas(32) double temp1[skip];
                alignas(32) double temp2[skip];

                _mm256_store_pd(temp1, result1);
                _mm256_store_pd(temp2, result2);

                partial_1 += temp1[0] + temp1[1] + temp1[2] + temp1[3];
                partial_2 += temp2[0] + temp2[1] + temp2[2] + temp2[3];
            }

            dest[row_start] = dest[row_start] * beta + partial_1 * alpha;
            dest[row_start + 1] = dest[row_start + 1] * beta + partial_2 * alpha;
        }


#pragma omp parallel for default(none) shared(std::cout, alpha, beta, matrix, vector, dest, rows, row_count, columns) private(start) schedule(dynamic)
        for (start = row_count; start < rows; ++start) {
            double partial = 0;


#pragma omp simd reduction(+:partial) aligned(vector: 32) aligned(matrix: 32)
            for (size_t i = 0; i < columns; ++i) {
                partial += vector[i] * matrix[start * columns + i];
            }

            dest[start] = dest[start] * beta + partial * alpha;
        }
    }

#define BLOCK_SIZE 1024

    template<>
    void gemv_manual2<double>(
        const double *matrix,
        const double *vector,
        double *dest,
        double const alpha,
        double const beta,
        size_t const rows,
        size_t const columns) {

        constexpr size_t vector_length{get_block_len<double>()};
        constexpr auto reduction_levels = static_cast<size_t>(std::log2(BLOCK_SIZE / vector_length));
        constexpr size_t reduction_count = BLOCK_SIZE / vector_length;
        const size_t row_blocks = (rows / BLOCK_SIZE) + (rows % BLOCK_SIZE > 0);
        const size_t column_blocks = (columns / BLOCK_SIZE);


#ifndef BENCHMARK
        assert(BLOCK_SIZE % vector_length == 0);
        assert(static_cast<size_t>(std::pow(2, reduction_levels)) == (BLOCK_SIZE / vector_length));
        std::cout << "total column blocks: "<< column_blocks << std::endl;
        std::cout << "total row blocks:  "<< row_blocks << std::endl;
        std::cout << "reduction count:  "<< reduction_count << std::endl;
#endif

        // multiple y by Beta
        for (size_t i = 0; i < rows; ++i) {
            dest[i] *= beta;
        }

        for (size_t row_block = 0; row_block < row_blocks; ++row_block) {
            size_t column_block = 0;

            const size_t block_end{row_block * BLOCK_SIZE + BLOCK_SIZE < rows ? row_block * BLOCK_SIZE + BLOCK_SIZE : rows};

            // std::cout << "row block " << row_block * BLOCK_SIZE << std::endl;
            // std::cout << "row end " << block_end << std::endl;

            for (; column_block < column_blocks * BLOCK_SIZE; column_block += BLOCK_SIZE) {
                // std::cout << column_block << " " << column_blocks * BLOCK_SIZE << std::endl;
                size_t i = 0;

                // std::cout << "-------------------------------------" << std::endl;

#pragma omp parallel for default(none) shared(reduction_levels, row_block, reduction_count, column_block, block_end, std::cout, alpha, beta, matrix, vector, dest, rows, columns) private(i) schedule(dynamic)
                for (i = row_block * BLOCK_SIZE; i < block_end; ++i) {
                    // std::cout << "--------------------" << std::endl;
                    // std::cout << i << std::endl;
                    __m256d reductions[reduction_count]{};
                    size_t current_reduction = 0;

                    for (size_t j = column_block; j < column_block + BLOCK_SIZE; j += vector_length) {
                        __m256d const vector_block = _mm256_load_pd(&vector[j]);
                        __m256d const mat_block = _mm256_loadu_pd(&matrix[i * columns + j]);

                        reductions[current_reduction] = _mm256_mul_pd(vector_block, mat_block);

                        ++current_reduction;
                        // std::cout << (current_reduction > reduction_count) << std::endl;
                    }

                    size_t reduction_count2 {reduction_count};

                    for (size_t levels = reduction_levels; levels > 0; --levels) {
                        for (size_t r = 0; r < reduction_count2 / 2; ++r) {
                            reductions[r] = _mm256_add_pd(reductions[r * 2], reductions[r * 2 + 1]);
                        }

                        reduction_count2 /= 2;
                    }
                    alignas(32) double temp[4];
                    _mm256_store_pd(temp, reductions[0]);
                    dest[i] += (temp[0] + temp[1] + temp[2] + temp[3]) * alpha;

                }

            }

            size_t j = column_blocks * BLOCK_SIZE;

            // leftover columns
            // TODO: fix up by less than 4 edge case

            // std::cout << j << "\n" << std::endl;

            if (column_block != columns) {
                size_t col_end = columns - (columns % vector_length);
                size_t row;

                // std::cout << "rows: " << row_block * BLOCK_SIZE <<  " " << block_end <<  "\n" << std::endl;
                // std::cout << "cols: " << j <<  " " << col_end <<  "\n" << std::endl;

#pragma omp parallel for default(none) shared(row_block, reduction_count, col_end, j, column_block, block_end, std::cout, alpha, beta, matrix, vector, dest, rows, columns) private(row) schedule(dynamic)
                for (row = row_block * BLOCK_SIZE; row < block_end; ++row) {

                    double closest2 = std::ceil(std::log2((double)(columns - j) / (double)vector_length));
                    auto reduction_count2 = static_cast<size_t>(std::pow(2, closest2));
                    __m256d reductions[reduction_count]{};
                    auto reduction_levels2 = static_cast<size_t>(closest2);

                    // std::cout << std::endl;
                    // std::cout << "closest2: " <<  std::ceil((double)(columns - j) / (double)vector_length) << std::endl;
                    // std::cout << "closest2: " <<  closest2 << std::endl;
                    // std::cout << "reduction_count: " <<  reduction_count2 << std::endl;
                    // std::cout << "reduction level: " << reduction_levels2 << std::endl;

                    size_t current_reduction = 0;

                    alignas(32) double temp[4];

                    size_t pos;
                    for (pos = j; pos < col_end; pos += vector_length) {
                        __m256d const vector_block = _mm256_load_pd(&vector[pos]);
                        __m256d const mat_block = _mm256_loadu_pd(&matrix[row * columns + pos]);
                        reductions[current_reduction] = _mm256_mul_pd(vector_block, mat_block);
                        ++current_reduction;

                        // _mm256_store_pd(temp, vector_block);
                        // std::cout << "vector: " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << std::endl;

                        // _mm256_store_pd(temp, mat_block);
                        // std::cout << "matrix: " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << std::endl;
                    }

                    // std::cout << pos << std::endl;

                    _mm256_store_pd(temp, reductions[0]);
                    // std::cout << "result " << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << std::endl;

                    double partial = 0;
                    for (; pos < columns; ++pos) {
                        // std::cout << "here " << std::endl;
                        partial += vector[pos] * matrix[row * columns + pos];
                        // std::cout << vector[pos] << " " << matrix[row * columns + pos] << " " << vector[pos] * matrix[row * columns + pos]  << " " << partial <<  std::endl;
                    }

                    // std::cout << "partial: " <<  partial << std::endl;


                    dest[row] += partial * alpha;

                    for (size_t levels = reduction_levels2; levels > 0; --levels) {
                        for (size_t r = 0; r < reduction_count2 / 2; ++r) {
                            reductions[r] = _mm256_add_pd(reductions[r * 2], reductions[r * 2 + 1]);
                        }
                        reduction_count2 /= 2;
                    }

                    // _mm256_store_pd(temp, reductions[0]);
                    // std::cout << temp[0] << " " << temp[1] << " " << temp[2] << " " << temp[3] << std::endl;

                    // alignas(32) double temp[4];
                    _mm256_store_pd(temp, reductions[0]);
                    dest[row] += (temp[0] + temp[1] + temp[2] + temp[3]) * alpha;
                }
            }
        }
    }

#endif
}
