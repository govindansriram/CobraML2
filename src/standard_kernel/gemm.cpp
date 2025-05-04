#include "cuda_math.h"
#include "standard_math.h"

namespace cobraml::core {

    template<typename T>
    static void gemm_naive(
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

        for (size_t i{0}; i < mat_one_rows; ++i) {
            for (size_t j{0}; j < mat_two_columns; ++j) {
                matrix_dest[row_stride_dest * i + j] *= beta;
                T acum{0};
                for (size_t k{0}; k < shared; ++k)
                    acum += static_cast<int8_t>(matrix_one[row_stride_one * i + k] * matrix_two[row_stride_two * k + j]);

                acum *= alpha;
                matrix_dest[row_stride_dest * i + j] += acum;
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
                gemm_naive(
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
            default: {
                throw std::runtime_error("invalid function provided");
            }
        }
#else
        gemm_naive(
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
#endif
    }

    void StandardMath::gemm(
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
