//
// Created by Sriram Govindan on 12/23/24.
//

#include "standard_math.h"
#include <omp.h>
#include "enums.h"
#include <cstring>
#include <iostream>

#ifdef AVX2
#include <immintrin.h>
#endif

#ifndef BENCHMARK
#include <cassert>
#endif

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
        size_t const row_count{get_row_count(rows, SKIP)}; // get rows w/o remainders
        constexpr size_t skip{get_block_len<float>()}; // SIMD vector length for float dtype
        constexpr size_t jump{UNROLLS * skip}; // when unrolled multiple SIMD operations are conducted this number covers
        // the amount
        const size_t column_count{columns / jump}; // the amount of columns interacted with

#ifndef BENCHMARK
        assert(reinterpret_cast<uintptr_t>(matrix) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(vector) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(dest) % 32 == 0);

        // std::cout << "col count " << column_count << std::endl;;
        // std::cout << "columns " << columns << std::endl;
        // std::cout << "row count " << row_count << std::endl;;
        // std::cout << "rows " << rows << std::endl;
#endif

#pragma omp parallel for default(none) shared(std::cout, column_count, alpha, beta, matrix, vector, dest, row_count, columns, skip) private(start) schedule(dynamic)
        for (start = 0; start < row_count / SKIP; ++start) {
            __m256 partial_1 = _mm256_setzero_ps();
            __m256 partial_2 = _mm256_setzero_ps();

            const size_t row_start{start * SKIP};

            for (size_t i = 0; i < column_count; i += 1) {
                __m256 const vector_block1{_mm256_load_ps(&vector[i * jump])};
                __m256 const vector_block2{_mm256_load_ps(&vector[i * jump + skip])};

                __m256 const mat_block_1_1{_mm256_loadu_ps(&matrix[row_start * columns + (i * jump)])};
                __m256 const mat_block_1_2{_mm256_loadu_ps(&matrix[row_start * columns + (i * jump) + skip])};

                __m256 const mat_block_2_1{_mm256_loadu_ps(&matrix[(row_start + 1) * columns + (i * jump)])};
                __m256 const mat_block_2_2{_mm256_loadu_ps(&matrix[(row_start + 1) * columns + (i * jump) + skip])};

                partial_1 = _mm256_fmadd_ps(vector_block1, mat_block_1_1, partial_1);
                partial_1 = _mm256_fmadd_ps(vector_block2, mat_block_1_2, partial_1);

                partial_2 = _mm256_fmadd_ps(vector_block1, mat_block_2_1, partial_2);
                partial_2 = _mm256_fmadd_ps(vector_block2, mat_block_2_2, partial_2);
            }

            // cleanup remainders
            for (size_t i = column_count * jump; i < columns; i += skip) {
                __m256 mat_block_1;
                __m256 mat_block_2;
                __m256 const vector_block{_mm256_load_ps(&vector[i])};

                if (i + skip > columns) {
                    size_t const amt = columns - i;
                    alignas(32) float block_1[skip]{};
                    alignas(32) float block_2[skip]{};

                    std::memcpy(block_1, &matrix[row_start * columns + i], amt * 4);
                    std::memcpy(block_2, &matrix[(row_start + 1) * columns + i], amt * 4);

                    mat_block_1 = _mm256_load_ps(block_1);
                    mat_block_2 = _mm256_load_ps(block_2);
                } else {
                    mat_block_1 = _mm256_loadu_ps(&matrix[row_start * columns + i]);
                    mat_block_2 = _mm256_loadu_ps(&matrix[(row_start + 1) * columns + i]);
                }

                partial_1 = _mm256_fmadd_ps(vector_block, mat_block_1, partial_1);
                partial_2 = _mm256_fmadd_ps(vector_block, mat_block_2, partial_2);
            }

            alignas(32) float temp1[skip];
            alignas(32) float temp2[skip];

            _mm256_store_ps(temp1, partial_1);
            _mm256_store_ps(temp2, partial_2);

            dest[row_start] = dest[row_start] * beta + (temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp1[4] + temp1[5] + temp1[6] + temp1[7]) * alpha;
            dest[row_start + 1] = dest[row_start + 1] * beta + (temp2[0] + temp2[1] + temp2[2] + temp2[3] + temp2[4] + temp2[5] + temp2[6] + temp2[7]) * alpha;
        }


#pragma omp parallel for default(none) shared(std::cout, alpha, beta, matrix, vector, dest, rows, row_count, columns) private(start) schedule(dynamic)
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
        size_t const row_count{get_row_count(rows, SKIP)}; // get rows w/o remainders
        constexpr size_t skip{get_block_len<double>()}; // SIMD vector length for double dtype
        constexpr size_t jump{UNROLLS * skip}; // when unrolled multiple SIMD operations are conducted this number covers
        // the amount
        const size_t column_count{columns / jump}; // the amount of columns interacted with

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
#endif
}
