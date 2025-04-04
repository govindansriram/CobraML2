//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include <iostream>
#include <cmath>
#include "../math_dis.h"

namespace cobraml::core {
    template<typename NumType>
    void gemv_naive(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns,
        size_t const row_stride) {
        for (size_t start{0}; start < rows; ++start) {
            NumType partial = 0;
            for (size_t i = 0; i < columns; ++i) {
                partial = static_cast<NumType>(partial + vector[i] * matrix[start * row_stride + i]);
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
        const size_t columns,
        size_t const row_stride) {
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns, row_stride) private(start) schedule(dynamic)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;

            for (size_t i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * row_stride + i]);
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
        const size_t columns,
        size_t const row_stride) {
        size_t start;

#pragma omp parallel for default(none) shared(alpha, beta, matrix, vector, dest, rows, columns, row_stride) private(start) schedule(static)
        for (start = 0; start < rows; ++start) {
            NumType partial = 0;

            size_t i;
#pragma omp simd reduction(+:partial)
            for (i = 0; i < columns; ++i) {
                partial += static_cast<NumType>(vector[i] * matrix[start * row_stride + i]);
            }

            dest[start] = static_cast<NumType>(dest[start] * beta + partial * alpha);
        }
    }

    template<typename Numtype>
    constexpr size_t get_block_len() {
        return 256 / (dtype_to_bytes(get_dtype_from_type<Numtype>::type) * 8);
    }

    inline size_t get_row_count(size_t const rows, size_t const skip) {
        size_t const remainder = rows % skip;
        size_t const row_count = rows - remainder;
        return row_count;
    }


    template<typename NumType>
    void gemv_manual(
        const NumType *matrix,
        const NumType *vector,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        const size_t rows,
        const size_t columns,
        size_t const row_stride) {
        gemv_parallel_simd(matrix, vector, dest, alpha, beta, rows, columns, row_stride);
    }

    template<typename NumType>
    void hadamard_product_naive(
        const NumType *tensor_one,
        const NumType *tensor_two,
        NumType *tensor_dest,
        const size_t rows,
        const size_t columns,
        const size_t row_stride) {
        size_t row;
#pragma omp parallel for default(none) shared(tensor_one, tensor_two, tensor_dest, rows, columns, row_stride) private(row)
        for (row = 0; row < rows; ++row) {
#pragma omp simd
            for (size_t column = 0; column < columns; ++column) {
                tensor_dest[row * row_stride + column] =
                        tensor_one[row * row_stride + column] * tensor_two[row * row_stride + column];
            }
        }
    }

    template<typename NumType>
    void element_wise_power(
        const NumType *tensor_one,
        const NumType *tensor_exponent,
        NumType *tensor_dest,
        const size_t rows,
        const size_t columns,
        const size_t row_stride) {
        size_t row;
#pragma omp parallel for default(none) shared(tensor_one, tensor_exponent, tensor_dest, rows, columns, row_stride) private(row)
        for (row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                tensor_dest[row * row_stride + column] = static_cast<NumType>(
                    pow(static_cast<double>(tensor_one[row * row_stride + column]),
                        static_cast<double>(tensor_exponent[row * row_stride + column])));
            }
        }
    }

    template<typename NumType>
    void element_wise_add(
        const NumType *tensor_one,
        const NumType *tensor_two,
        NumType *tensor_dest,
        const size_t rows,
        const size_t columns,
        const size_t row_stride) {
        size_t row;
#pragma omp parallel for default(none) shared(tensor_one, tensor_two, tensor_dest, rows, columns, row_stride) private(row)
        for (row = 0; row < rows; ++row) {
#pragma omp simd
            for (size_t column = 0; column < columns; ++column) {
                tensor_dest[row * row_stride + column] =
                        tensor_one[row * row_stride + column] + tensor_two[row * row_stride + column];
            }
        }
    }

    template<typename NumType>
void element_wise_subtract(
    const NumType *tensor_one,
    const NumType *tensor_two,
    NumType *tensor_dest,
    const size_t rows,
    const size_t columns,
    const size_t row_stride) {
        size_t row;
#pragma omp parallel for default(none) shared(tensor_one, tensor_two, tensor_dest, rows, columns, row_stride) private(row)
        for (row = 0; row < rows; ++row) {
#pragma omp simd
            for (size_t column = 0; column < columns; ++column) {
                tensor_dest[row * row_stride + column] =
                        tensor_one[row * row_stride + column] - tensor_two[row * row_stride + column];
            }
        }
    }

    template<>
    void element_wise_power<double>(
        const double *tensor_one,
        const double *tensor_two,
        double *tensor_dest,
        size_t rows,
        size_t columns,
        size_t row_stride);

    template<typename T>
    using ElementWiseFunc = void (*)(const T *, const T *, T *, size_t, size_t, size_t);

    template<typename T>
    ElementWiseFunc<T> element_wise_func_dispatcher(uint8_t index) {
        switch (index) {
            case 0:
                return &hadamard_product_naive<T>; // Return a pointer to func_0
            case 1:
                return &element_wise_power<T>; // Return a pointer to func_1
            case 2:
                return &element_wise_add<T>;
            case 3:
                return &element_wise_subtract<T>;
            default:
                throw std::runtime_error("invalid index provided for element wise function");
        }
    }


#ifdef AVX2

    template<>
    void gemv_manual<float>(
        const float *matrix,
        const float *vector,
        float *dest,
        float alpha,
        float beta,
        size_t rows,
        size_t columns,
        size_t row_stride);

    template<>
    void gemv_manual<double>(
        const double *matrix,
        const double *vector,
        double *dest,
        double alpha,
        double beta,
        size_t rows,
        size_t columns,
        size_t row_stride);

#endif

#ifdef BENCHMARK

    template<typename NumType>
    void benchmarked_gemv(
        const NumType *mat,
        const NumType *vec,
        NumType *dest,
        const NumType alpha,
        const NumType beta,
        size_t const rows,
        size_t const columns,
        size_t const row_stride) {
        switch (func_pos) {
            case 0: {
                gemv_naive(mat, vec, dest, alpha, beta, rows, columns, row_stride);
                return;
            }
            case 1: {
                gemv_parallel(mat, vec, dest, alpha, beta, rows, columns, row_stride);
                return;
            }
            case 2: {
                gemv_parallel_simd(mat, vec, dest, alpha, beta, rows, columns, row_stride);
                return;
            }
            case 3: {
                gemv_manual(mat, vec, dest, alpha, beta, rows, columns, row_stride);
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
        size_t const columns,
        size_t const row_stride) {
        gemv_manual(mat, vec, dest, alpha, beta, rows, columns, row_stride);
    }
#endif

    class StandardMath final : public Math {
    public:
        void gemv(
            const void *matrix,
            const void *vector,
            void *dest,
            const void *alpha,
            const void *beta,
            size_t rows,
            size_t columns,
            size_t row_stride,
            Dtype dtype) override;

        void hadamard_product(
            const void *tensor_one,
            const void *tensor_two,
            void *tensor_dest,
            size_t rows,
            size_t columns,
            size_t row_stride,
            Dtype dtype) override;

        void element_wise_power(
            const void *tensor_one,
            const void *exponent_tensor,
            void *tensor_dest,
            size_t rows,
            size_t columns,
            size_t row_stride,
            Dtype dtype) override;

        void element_wise_add(
            const void *tensor_one,
            const void *tensor_two,
            void *tensor_dest,
            size_t rows,
            size_t columns,
            size_t row_stride,
            Dtype dtype) override;

        void element_wise_sub(
            const void *tensor_one,
            const void *tensor_two,
            void *tensor_dest,
            size_t rows,
            size_t columns,
            size_t row_stride,
            Dtype dtype) override;
    };
}

#endif //STANDARD_MATH_H
