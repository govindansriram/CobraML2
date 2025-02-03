//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include <iostream>
#include <immintrin.h>
#include "../math_dis.h"

namespace cobraml::core {
    void set_num_threads();

    template<typename NumType>
    void gemv_naive(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        for (size_t start{0}; start < rows; ++start) {
            NumType partial = 0;
            for (size_t i = 0; i < columns; ++i) {
                partial = static_cast<NumType>(partial + vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

    template<typename NumType>
    void gemv_parallel(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns) private(start) schedule(dynamic)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;

            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

    template<typename NumType>
    void gemv_parallel_simd(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns) private(start) schedule(dynamic)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;

#pragma omp simd reduction(+:partial)
            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

#define ROW_COUNT 4

    template<typename NumType>
    void gemv_parallel_simd_2(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {
        set_num_threads();
        size_t start;
        size_t const remainder = rows % ROW_COUNT;
        size_t const row_count = rows - remainder;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, row_count, columns) private(start) schedule(dynamic)
        for (start = 0; start < row_count; start += ROW_COUNT) {
            NumType partial = 0;
            NumType partial_2 = 0;
            NumType partial_3 = 0;
            NumType partial_4 = 0;

#pragma omp simd reduction(+:partial) reduction(+:partial_2) reduction(+:partial_3) reduction(+:partial_4) aligned(vector: 32) aligned(matrix: 32)
            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
                partial_2 += static_cast<NumType>(vector[i] * matrix[(start + 1) * columns + i]);
                partial_3 += static_cast<NumType>(vector[i] * matrix[(start + 2) * columns + i]);
                partial_4 += static_cast<NumType>(vector[i] * matrix[(start + 3) * columns + i]);

                // _mm_prefetch(&matrix[(start + 320)], _MM_HINT_T0);
                // _mm_prefetch(&matrix[(start + 320)], _MM_HINT_T0);
                // _mm_prefetch(&matrix[(start + 320)], _MM_HINT_T0);
                // _mm_prefetch(&matrix[(start + 320)], _MM_HINT_T0);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
            dest[start + 1] = static_cast<NumType>(dest[start + 1] * beta + partial_2 * alpha);
            dest[start + 2] = static_cast<NumType>(dest[start + 2] * beta + partial_3 * alpha);
            dest[start + 3] = static_cast<NumType>(dest[start + 3] * beta + partial_4 * alpha);
        }


#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, row_count, columns) private(start) schedule(dynamic)
        for (start = row_count; start < rows; ++start) {
            NumType partial = 0;

#pragma omp simd reduction(+:partial) aligned(vector: 32) aligned(matrix: 32)
            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * columns + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

    template <typename Numtype>
    constexpr size_t get_block_len() {
        return 256 / (dtype_to_bytes(get_dtype_from_type<Numtype>::type) * 8);
    }

#define ROW_COUNT2 2

    void float32_gemv_kernel(const float *matrix,
                             const float *vector,
                             float *dest,
                             float const &alpha,
                             float const &beta,
                             size_t const &rows,
                             size_t const &columns,
                             size_t const &row_count);

    void float64_gemv_kernel(const double *matrix,
                             const double *vector,
                             double *dest,
                             double const &alpha,
                             double const &beta,
                             size_t const &rows,
                             size_t const &columns,
                             size_t const &row_count);


    template<typename NumType>
    void gemv_parallel_simd_3(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns) {

        set_num_threads();
        size_t const remainder = rows % ROW_COUNT2;
        size_t const row_count = rows - remainder;

        if constexpr (std::is_same_v<NumType, float>){
            float32_gemv_kernel(matrix, vector, dest, alpha, beta, rows, columns, row_count);
            return;
        }

        if constexpr (std::is_same_v<NumType, double>){
            float64_gemv_kernel(matrix, vector, dest, alpha, beta, rows, columns, row_count);
            return;
        }

        gemv_parallel_simd(matrix, vector, dest, alpha, beta, rows, columns);

        // switch (get_dtype_from_type<NumType>::type) {
        //     case FLOAT32: {
        //         float32_gemv_kernel(matrix, vector, dest, alpha, beta, rows, columns, row_count);
        //         return;
        //     }
        //     case FLOAT64: {
        //         float64_gemv_kernel(matrix, vector, dest, alpha, beta, rows, columns, row_count);
        //         return;
        //     }
        //     default:
        //         gemv_parallel_simd(matrix, vector, dest, alpha, beta, rows, columns);
        // }

    }


#ifdef BENCHMARK

    template<typename NumType>
    void benchmarked_gemv(
        const NumType *mat,
        const NumType *vec,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        size_t const rows,
        size_t const columns) {
        switch (func_pos) {
            case 0: {
                gemv_naive(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 1: {
                gemv_parallel(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 2: {
                gemv_parallel_simd(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 3: {
                gemv_parallel_simd_2(mat, vec, dest, alpha, beta, rows, columns);
                return;
            }
            case 4: {
                gemv_parallel_simd_3(mat, vec, dest, alpha, beta, rows, columns);
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
        const NumType alpha,
        const NumType beta,
        size_t const rows,
        size_t const columns) {
        gemv_parallel_simd_3(mat, vec, dest, alpha, beta, rows, columns);
    }
#endif

    class StandardMath final : public Math {
        void gemv(
            const void *matrix,
            const void *vector,
            void *dest,
            const void *alpha,
            const void *beta,
            size_t rows,
            size_t columns,
            Dtype dtype) override;
    };
}

#endif //STANDARD_MATH_H
