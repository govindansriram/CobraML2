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

    static size_t get_thread_count() {
        set_num_threads();
        static size_t thread_count{0};

        if (!thread_count) {
#ifdef NUM_THREADS
            thread_count = NUM_THREADS;
#else
            thread_count = omp_get_max_threads();
#endif
        }

        return thread_count;
    }

    TensorIter::TensorIter(const size_t *shape, const size_t *stride_one, const size_t *stride_two,
                           const size_t shape_len): shape(shape), stride_one(stride_one), stride_two(stride_two),
                                                    shape_len(shape_len) {
        computed_stride.reserve(shape_len);
        size_t sum{1};
        for (size_t i{0}; i < shape_len; ++i) {
            sum *= shape[i];
        }

        total_elements = sum;

        for (size_t i{0}; i < shape_len; ++i) {
            sum /= shape[i];
            computed_stride.push_back(sum);
        }

#ifndef BENCHMARK
        assert(computed_stride[computed_stride.size() - 1] == 1);
#endif
    }

    inline size_t compute_index(size_t index, const std::vector<size_t> &generic_stride, const size_t *padded_stride) {
        size_t comp_ind{0};

        for (size_t i{0}; i < generic_stride.size(); ++i) {
            const size_t mult = index / generic_stride[i];
            comp_ind += mult * padded_stride[i];
            index = index % generic_stride[i];
        }

        return comp_ind;
    }

    void TensorIter::get_indexes(size_t *index_buffer_1, size_t *index_buffer_2, const size_t start_index,
                                 const size_t index_count) const {
        if (start_index + index_count > total_elements) {
            throw std::runtime_error("requested element not present in the array");
        }

        for (size_t i{0}; i < index_count; ++i) {
            const auto idx{start_index + i};
            index_buffer_1[i] = compute_index(idx, computed_stride, stride_one);
            index_buffer_2[i] = compute_index(idx, computed_stride, stride_two);
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
        size_t const row_stride,
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
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns, row_stride);
                return;
            }
            case FLOAT32: {
                const auto casted_dest = static_cast<float *>(dest);
                const auto casted_mat = static_cast<const float *>(matrix);
                const auto casted_vec = static_cast<const float *>(vector);
                const auto casted_alpha = static_cast<const float *>(alpha);
                const auto casted_beta = static_cast<const float *>(beta);
                benchmarked_gemv<float>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns, row_stride);
                return;
            }
            case INT8: {
                const auto casted_dest = static_cast<int8_t *>(dest);
                const auto casted_mat = static_cast<const int8_t *>(matrix);
                const auto casted_vec = static_cast<const int8_t *>(vector);
                const auto casted_alpha = static_cast<const int8_t *>(alpha);
                const auto casted_beta = static_cast<const int8_t *>(beta);
                benchmarked_gemv<int8_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns, row_stride);
                return;
            }
            case INT16: {
                const auto casted_dest = static_cast<int16_t *>(dest);
                const auto casted_mat = static_cast<const int16_t *>(matrix);
                const auto casted_vec = static_cast<const int16_t *>(vector);
                const auto casted_alpha = static_cast<const int16_t *>(alpha);
                const auto casted_beta = static_cast<const int16_t *>(beta);
                benchmarked_gemv<int16_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns, row_stride);
                return;
            }
            case INT32: {
                const auto casted_dest = static_cast<int32_t *>(dest);
                const auto casted_mat = static_cast<const int32_t *>(matrix);
                const auto casted_vec = static_cast<const int32_t *>(vector);
                const auto casted_alpha = static_cast<const int32_t *>(alpha);
                const auto casted_beta = static_cast<const int32_t *>(beta);
                benchmarked_gemv<int32_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns, row_stride);
                return;
            }
            case INT64: {
                const auto casted_dest = static_cast<int64_t *>(dest);
                const auto casted_mat = static_cast<const int64_t *>(matrix);
                const auto casted_vec = static_cast<const int64_t *>(vector);
                const auto casted_alpha = static_cast<const int64_t *>(alpha);
                const auto casted_beta = static_cast<const int64_t *>(beta);
                benchmarked_gemv<int64_t>(
                    casted_mat, casted_vec, casted_dest, *casted_alpha, *casted_beta, rows, columns, row_stride);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate gemv on invalid type");
            }
        }
    }

#ifndef BENCHMARK
#define BLOCK 5
#else
#define BLOCK 1000
#endif

#define MULT_OP(multiplier, multiplicand) ((multiplier) * (multiplicand))
#define DIV_OP(multiplier, multiplicand) ((multiplier) / (multiplicand))
#define ADD_OP(multiplier, multiplicand) ((multiplier) + (multiplicand))
#define SUB_OP(multiplier, multiplicand) ((multiplier) - (multiplicand))

#define OMP_PARALLEL_FOR_ELEMENT_WISE _Pragma("omp parallel for default(none) shared(std::cout, typed_tensor_one, typed_tensor_two, typed_tensor_dest, shape, shape_len, stride_one, stride_two, dest_row_stride, max_iters, iter) private(cur_iter)")
#define OMP_SIMD_FOR_ELEMENT_WISE _Pragma("omp simd")

    // TODO preallocate data_ptr outside loop
    /**
     * Iterates through two tensors with the same "shape" and performs an element wise operation
     * @param operation
     * @param tensor_one
     * @param tensor_two
     * @param tensor_dest
     * @param shape
     * @param shape_len
     * @param stride_one
     * @param stride_two
     * @param dest_row_stride
     * @param max_iters
     * @param iter
     */
#define ELEMENT_WISE_ITERATOR(operation, tensor_one, tensor_two, tensor_dest, shape, shape_len, stride_one, stride_two, dest_row_stride, max_iters, iter){\
    const TensorIter iter(shape, stride_one, stride_two, shape_len);\
    const auto max_iters{static_cast<size_t>(std::ceil(static_cast<float>(iter.total_elements) / BLOCK))};\
    size_t cur_iter;\
OMP_PARALLEL_FOR_ELEMENT_WISE\
    for (cur_iter = 0; cur_iter < max_iters; ++cur_iter) {\
        const size_t start_index{cur_iter * BLOCK};\
        const size_t total{start_index + BLOCK > iter.total_elements ? iter.total_elements - start_index : BLOCK};\
        auto *data_ptr = new size_t[total * 2];\
        iter.get_indexes(data_ptr, data_ptr + total, start_index, total);\
OMP_SIMD_FOR_ELEMENT_WISE\
        for (size_t idx = 0; idx < total; ++idx) {\
            const size_t dest_index{((cur_iter * BLOCK + idx) / shape[shape_len - 1] * dest_row_stride) + (\
            (cur_iter * BLOCK + idx) % shape[shape_len - 1])};\
            tensor_dest[dest_index] = operation(tensor_one[data_ptr[idx]], tensor_two[data_ptr[total + idx]]);\
        }\
        delete[] data_ptr;\
    }\
}

    template<typename NumType>
    void element_wise_op_handler(
        const Operations op,
        const void *tensor_one,
        const void *tensor_two,
        void *tensor_dest,
        const size_t *shape,
        const size_t shape_len,
        const size_t *stride_one,
        const size_t *stride_two,
        const size_t dest_row_stride) {
        const auto typed_tensor_one{static_cast<const NumType *>(tensor_one)};
        const auto typed_tensor_two{static_cast<const NumType *>(tensor_two)};
        const auto typed_tensor_dest{static_cast<NumType *>(tensor_dest)};

        switch (op) {
            case MULT: {
                ELEMENT_WISE_ITERATOR(MULT_OP, typed_tensor_one, typed_tensor_two, typed_tensor_dest, shape, shape_len,
                                      stride_one, stride_two, dest_row_stride, max_iters, iter);
                return;
            }
            case ADD: {
                ELEMENT_WISE_ITERATOR(ADD_OP, typed_tensor_one, typed_tensor_two, typed_tensor_dest, shape, shape_len,
                                      stride_one, stride_two, dest_row_stride, max_iters, iter);
                return;
            }
            default: {
                throw std::runtime_error("invalid element wise operation specified");
            }
        }
    }

    void element_wise_type_handler(
        const Operations op,
        const void *buffer_one,
        const void *buffer_two,
        void *buffer_dest,
        const size_t *shape,
        const size_t shape_len,
        const size_t *stride_one,
        const size_t *stride_two,
        const size_t dest_row_stride,
        const Dtype dtype) {
        switch (dtype) {
            case FLOAT64: {
                element_wise_op_handler<double>(op, buffer_one, buffer_two,
                                                buffer_dest, shape, shape_len, stride_one, stride_two, dest_row_stride);
                return;
            }
            case FLOAT32: {
                element_wise_op_handler<float>(op, buffer_one, buffer_two,
                                               buffer_dest, shape, shape_len, stride_one, stride_two, dest_row_stride);
                return;
            }
            case INT8: {
                element_wise_op_handler<int8_t>(op, buffer_one, buffer_two,
                                                buffer_dest, shape, shape_len, stride_one, stride_two, dest_row_stride);
                return;
            }
            case INT16: {
                element_wise_op_handler<int16_t>(op, buffer_one, buffer_two,
                                                 buffer_dest, shape, shape_len, stride_one, stride_two,
                                                 dest_row_stride);
                return;
            }
            case INT32: {
                element_wise_op_handler<int32_t>(op, buffer_one, buffer_two,
                                                 buffer_dest, shape, shape_len, stride_one, stride_two,
                                                 dest_row_stride);
                return;
            }
            case INT64: {
                element_wise_op_handler<int64_t>(op, buffer_one, buffer_two,
                                                 buffer_dest, shape, shape_len, stride_one, stride_two,
                                                 dest_row_stride);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate gemv on invalid type");
            }
        }
    }

    inline void compute_shape(
        size_t index,
        size_t * current_shape,
        const size_t * original_stride,
        const size_t shape_len) {

        for (size_t i{0}; i < shape_len; ++i) {
            current_shape[i] = index / original_stride[i];
            index -= original_stride[i] * current_shape[i];
        }
    }

    inline size_t calculate_index(
        const size_t * current_shape,
        const size_t * permute_mask,
        const size_t * new_stride,
        const size_t shape_len) {

        size_t idx{0};
        for (size_t i{0}; i < shape_len; ++i) idx += current_shape[permute_mask[i]] * new_stride[i];
        return idx;
    }

    template<typename Dtype>
    void basic_permutation(
        const void * tensor,
        void * dest,
        const size_t shape_len,
        const size_t row_stride,
        const size_t rows,
        const size_t columns,
        const size_t * permute_mask,
        const size_t * original_stride,
        const size_t * dest_stride) {

        // std::cout << original_stride[0] << original_stride[1] << std::endl;

        auto cast_tensor{static_cast<const Dtype *>(tensor)};
        auto cast_dest{static_cast<Dtype *>(dest)};

        // TODO create kernels for common case, matrix transpose, flatten in possible

        const size_t tc{get_thread_count()};
        auto shape_ptr = new size_t[tc * shape_len];

        size_t column;
        for (size_t row{0}; row < rows;  ++row) {
            // TODO add prefetching of some sort for transposed column variables
#pragma omp parallel for default(none) shared(std::cout, columns, row, row_stride, cast_tensor, cast_dest, shape_ptr, shape_len, permute_mask, dest_stride, original_stride) private(column)
            for (column = 0; column < columns; ++column) {
                size_t index = row * row_stride + column;
                Dtype val = cast_tensor[index];

                const size_t tid{static_cast<size_t>(omp_get_thread_num())};
                size_t * current_shape{shape_ptr + tid * shape_len};
                compute_shape(index, current_shape, original_stride, shape_len);

                // std::cout << current_shape[0] << current_shape[1] << std::endl;

                size_t dest_index = calculate_index(current_shape, permute_mask, dest_stride, shape_len);
                cast_dest[dest_index] = val;
            }
        }

        delete[] shape_ptr;
    }

    size_t total_rows(const size_t * shape, const size_t shape_len) {
        size_t rows{1};
        for (size_t i{0}; i < shape_len - 1; ++i) rows *= shape[i];
        return rows;
    }

    void StandardMath::permute(
        const void *tensor,
        void *dest,
        const size_t shape_len,
        const size_t *original_shape,
        const size_t *permute_mask,
        const size_t *original_stride,
        const size_t *dest_stride,
        const Dtype dtype) {

        const size_t row_stride{original_stride[shape_len - 2]};
        const size_t columns{original_shape[shape_len - 1]};
        const size_t rows{total_rows(original_shape, shape_len)};

        switch (dtype) {
            case FLOAT64: {
                basic_permutation<double>(tensor, dest, shape_len, row_stride, rows,
                    columns, permute_mask, original_stride, dest_stride);
                return;
            }
            case FLOAT32: {
                basic_permutation<float>(tensor, dest, shape_len, row_stride, rows,
                    columns, permute_mask, original_stride, dest_stride);
                return;
            }
            case INT64: {
                basic_permutation<int64_t>(tensor, dest, shape_len, row_stride, rows,
                    columns, permute_mask, original_stride, dest_stride);
                return;
            }
            case INT32: {
                basic_permutation<int32_t>(tensor, dest, shape_len, row_stride, rows,
                    columns, permute_mask, original_stride, dest_stride);
                return;
            }
            case INT16: {
                basic_permutation<int16_t>(tensor, dest, shape_len, row_stride, rows,
                    columns, permute_mask, original_stride, dest_stride);
                return;
            }
            case INT8:{
                basic_permutation<int8_t>(tensor, dest, shape_len, row_stride, rows,
                    columns, permute_mask, original_stride, dest_stride);
                return;
            }
            case INVALID: {
                throw std::runtime_error("cannot calculate gemv on invalid type");
            }
        }
    }


    // TODO uncomment this reduction algorithm
//     template<typename Dtype>
//     void single_reduction(
//         const Dtype * tensor,
//         Dtype * dest,
//         const size_t tensor_rows,
//         const size_t tensor_row_length,
//         const size_t tensor_row_stride,
//         const size_t dest_row_length,
//         const size_t dest_row_stride) {
//
//         // TODO add faster binary reduction kernel, add permutaiton code (transpose kernel)
//         // TODO add avx2 intrinsics
//         // TODO look into #pragma omp parallel for collapse(2)
//         // TODO look into omp tasks
//
//         const size_t tc = get_thread_count();
//         const size_t block_size{tc >= tensor_row_length ? tensor_row_length : tensor_row_length / tc};
//         const size_t block_count{static_cast<size_t>(
//                 std::ceil(static_cast<double>(tensor_row_length) / static_cast<double>(block_size)))};
//
//         auto partials = new Dtype[block_count]{};
//
//         size_t blok;
//         for (size_t row{0}; row < tensor_rows; ++row) {
//             size_t idx{row / dest_row_length * dest_row_stride + row % dest_row_length};
//
// #pragma omp parallel for default(none) shared(tensor, block_count, block_size, tensor_row_length, row, tensor_row_stride, dest, dest_row_length, dest_row_stride, idx, partials) private(blok)
//             for (blok = 0; blok < block_count; ++blok) {
//                 size_t start{blok * block_size};
//                 const size_t end = std::min((blok + 1) * block_size, tensor_row_length);
//                 Dtype sum{0};
//
// #pragma omp simd reduction(+:sum)
//                 for (; start < end; ++start) {
//                     sum += tensor[row * tensor_row_stride + start];
//                 }
//
//                 partials[blok] = sum;
//             }
//
//             Dtype sum{0};
// #pragma omp simd reduction(+:sum)
//             for (size_t i = 0; i < block_count; ++i) {
//                 sum += partials[i];
//             }
//
//             dest[idx] = sum;
//         }
//
//         delete[] partials;
//     }

    void StandardMath::hadamard_product(
        const void *tensor_one,
        const void *tensor_two,
        void *tensor_dest,
        const size_t *shape,
        const size_t shape_len,
        const size_t *stride_one,
        const size_t *stride_two,
        const size_t dest_row_stride,
        const Dtype dtype) {
        set_num_threads();

        element_wise_type_handler(
            MULT,
            tensor_one,
            tensor_two,
            tensor_dest,
            shape,
            shape_len,
            stride_one,
            stride_two,
            dest_row_stride,
            dtype);
    }

    void StandardMath::element_wise_add(
        const void *tensor_one,
        const void *tensor_two,
        void *tensor_dest,
        const size_t *shape,
        const size_t shape_len,
        const size_t *stride_one,
        const size_t *stride_two,
        const size_t dest_row_stride,
        const Dtype dtype) {
        set_num_threads();

        element_wise_type_handler(
            ADD,
            tensor_one,
            tensor_two,
            tensor_dest,
            shape,
            shape_len,
            stride_one,
            stride_two,
            dest_row_stride,
            dtype);
    }

#ifdef AVX2
#define SKIP 2
#define UNROLLS 2

#define SKIPf32 2
#define UNROLLf32 2

    template<>
    void gemv_manual<float>(
        const float *matrix,
        const float *vector,
        float *dest,
        float const alpha,
        float const beta,
        size_t const rows,
        size_t const columns,
        size_t const row_stride) {
        size_t start;
        size_t const row_count{get_row_count(rows, SKIPf32)}; // get rows w/o remainders
        constexpr size_t skip{get_block_len<float>()}; // SIMD vector length for float dtype
        constexpr size_t jump{UNROLLf32 * skip};
        // when unrolled multiple SIMD operations are conducted this number covers
        // the amount
        const size_t column_count{columns / jump}; // the amount of columns interacted with
        // std::cout << row_stride;

#ifndef BENCHMARK
        assert(reinterpret_cast<uintptr_t>(matrix) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(vector) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(dest) % 32 == 0);
#endif

#pragma omp parallel for default(none) shared(std::cout, column_count, alpha, beta, matrix, vector, dest, row_count, columns, skip, row_stride) private(start) schedule(static)
        for (start = 0; start < row_count / SKIPf32; ++start) {
            __m256 partial_1{_mm256_setzero_ps()};
            __m256 partial_2{_mm256_setzero_ps()};

            const size_t row_start{start * SKIPf32};

            for (size_t i = 0; i < column_count; i += 1) {
                __m256 const vector_block1{_mm256_load_ps(&vector[i * jump])};
                __m256 const vector_block2{_mm256_load_ps(&vector[i * jump + skip])};

                __m256 const mat_block_1_1{_mm256_load_ps(&matrix[row_start * row_stride + (i * jump)])};
                __m256 const mat_block_1_2{_mm256_load_ps(&matrix[row_start * row_stride + (i * jump) + skip])};

                __m256 const mat_block_2_1{_mm256_load_ps(&matrix[(row_start + 1) * row_stride + (i * jump)])};
                __m256 const mat_block_2_2{_mm256_load_ps(&matrix[(row_start + 1) * row_stride + (i * jump) + skip])};

                partial_1 = _mm256_fmadd_ps(vector_block1, mat_block_1_1, partial_1);
                partial_1 = _mm256_fmadd_ps(vector_block2, mat_block_1_2, partial_1);

                partial_2 = _mm256_fmadd_ps(vector_block1, mat_block_2_1, partial_2);
                partial_2 = _mm256_fmadd_ps(vector_block2, mat_block_2_2, partial_2);
            }

            // cleanup remainders
            for (size_t i = column_count * jump; i < columns; i += skip) {
                __m256 const vector_block{_mm256_load_ps(&vector[i])};

                const __m256 mat_block_1 = _mm256_load_ps(&matrix[row_start * row_stride + i]);
                const __m256 mat_block_2 = _mm256_load_ps(&matrix[(row_start + 1) * row_stride + i]);

                partial_1 = _mm256_fmadd_ps(vector_block, mat_block_1, partial_1);
                partial_2 = _mm256_fmadd_ps(vector_block, mat_block_2, partial_2);
            }

            alignas(32) float temp1[skip];
            alignas(32) float temp2[skip];

            _mm256_store_ps(temp1, partial_1);
            _mm256_store_ps(temp2, partial_2);

            dest[row_start] = dest[row_start] * beta + (
                                  temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp1[4] + temp1[5] + temp1[6] + temp1[7])
                              * alpha;
            dest[row_start + 1] = dest[row_start + 1] * beta + (
                                      temp2[0] + temp2[1] + temp2[2] + temp2[3] + temp2[4] + temp2[5] + temp2[6] + temp2
                                      [7]) * alpha;
        }


#pragma omp parallel for default(none) shared(std::cout, alpha, beta, matrix, vector, dest, rows, row_count, columns, row_stride) private(start) schedule(static)
        for (start = row_count; start < rows; ++start) {
            float partial = 0;


#pragma omp simd reduction(+:partial) aligned(vector: 32) aligned(matrix: 32)
            for (size_t i = 0; i < columns; ++i) {
                partial += vector[i] * matrix[start * row_stride + i];
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
        size_t const columns,
        size_t const row_stride) {
        size_t start;
        size_t const row_count{get_row_count(rows, SKIP)}; // get rows w/o remainders
        constexpr size_t skip{get_block_len<double>()}; // SIMD vector length for double dtype
        constexpr size_t jump{UNROLLS * skip};
        // when unrolled multiple SIMD operations are conducted this number covers
        // the amount
        const size_t column_count{columns / jump}; // the amount of columns interacted with
        // std::cout << row_stride;

#ifndef BENCHMARK
        assert(reinterpret_cast<uintptr_t>(matrix) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(vector) % 32 == 0);
        assert(reinterpret_cast<uintptr_t>(dest) % 32 == 0);
#endif

#pragma omp parallel for default(none) shared(row_stride, std::cout, column_count, alpha, beta, matrix, vector, dest, row_count, columns, skip) private(start) schedule(static)
        for (start = 0; start < row_count / SKIP; ++start) {
            double partial_1 = 0;
            double partial_2 = 0;

            const size_t row_start = start * SKIP;

            for (size_t i = 0; i < column_count; i += 1) {
                __m256d const vector_block1 = _mm256_load_pd(&vector[i * jump]);
                __m256d const vector_block2 = _mm256_load_pd(&vector[i * jump + skip]);

                __m256d const mat_block_1_1 = _mm256_load_pd(&matrix[row_start * row_stride + (i * jump)]);
                __m256d const mat_block_1_2 = _mm256_load_pd(&matrix[row_start * row_stride + (i * jump) + skip]);
                __m256d const mat_block_2_1 = _mm256_load_pd(&matrix[(row_start + 1) * row_stride + (i * jump)]);
                __m256d const mat_block_2_2 = _mm256_load_pd(&matrix[(row_start + 1) * row_stride + (i * jump) + skip]);

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
                __m256d const vector_block = _mm256_load_pd(&vector[i]);

                const __m256d mat_block_1 = _mm256_load_pd(&matrix[row_start * row_stride + i]);
                const __m256d mat_block_2 = _mm256_load_pd(&matrix[(row_start + 1) * row_stride + i]);

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


#pragma omp parallel for default(none) shared(row_stride, std::cout, alpha, beta, matrix, vector, dest, rows, row_count, columns) private(start) schedule(static)
        for (start = row_count; start < rows; ++start) {
            double partial = 0;


#pragma omp simd reduction(+:partial) aligned(vector: 32) aligned(matrix: 32)
            for (size_t i = 0; i < columns; ++i) {
                partial += vector[i] * matrix[start * row_stride + i];
            }

            dest[start] = dest[start] * beta + partial * alpha;
        }
    }
#endif
}
