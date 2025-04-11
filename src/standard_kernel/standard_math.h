//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_MATH_H
#define STANDARD_MATH_H

#include <iostream>
#include <cmath>
#include <vector>

#include "../math_dis.h"

namespace cobraml::core {

    enum Operations {
        MULT,
        DIV,
        ADD,
        SUB,
        POW,
    };

    struct TensorIter {
        std::vector<size_t> computed_stride{};
        const size_t *shape;
        const size_t *stride_one;
        const size_t *stride_two;
        size_t shape_len{0};
        size_t total_elements{0};

        TensorIter(const size_t *shape, const size_t *stride_one, const size_t *stride_two, size_t shape_len);

        TensorIter() = delete;

        TensorIter(const TensorIter &) = delete; // copy ctor
        TensorIter(TensorIter &&) = delete; // move ctor
        TensorIter &operator=(const TensorIter &) = delete; // copy assignment
        TensorIter &operator=(TensorIter &&) = delete;

        void get_indexes(
            size_t *index_buffer_1,
            size_t *index_buffer_2,
            size_t start_index,
            size_t index_count) const;
    };

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
            const size_t *shape,
            size_t shape_len,
            const size_t *stride_one,
            const size_t *stride_two,
            size_t dest_row_stride,
            Dtype dtype) override;

        void element_wise_add(
            const void *tensor_one,
            const void *tensor_two,
            void *tensor_dest,
            const size_t *shape,
            size_t shape_len,
            const size_t *stride_one,
            const size_t *stride_two,
            size_t dest_row_stride,
            Dtype dtype) override;

        // void element_wise_power(
        //     const void *tensor_one,
        //     const void *exponent_tensor,
        //     void *tensor_dest,
        //     size_t rows,
        //     size_t columns,
        //     size_t row_stride,
        //     Dtype dtype) override;
        //
        // void element_wise_add(
        //     const void *tensor_one,
        //     const void *tensor_two,
        //     void *tensor_dest,
        //     size_t rows,
        //     size_t columns,
        //     size_t row_stride,
        //     Dtype dtype) override;
        //
        // void element_wise_sub(
        //     const void *tensor_one,
        //     const void *tensor_two,
        //     void *tensor_dest,
        //     size_t rows,
        //     size_t columns,
        //     size_t row_stride,
        //     Dtype dtype) override;
    };
}

#endif //STANDARD_MATH_H
