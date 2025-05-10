//
// Created by sriram on 5/2/25.
//

#include "standard_math.h"

namespace cobraml::core {
        // TODO test equals and permute ensure gradient accumulation is possible by using these to run the reducion algorithm
    template<typename Dtype>
    void eq_kernel(
        const void *tensor_1,
        const void *tensor_2,
        int * result,
        const size_t total_len) {

        const auto t1 = static_cast<const Dtype *>(tensor_1);
        const auto t2 = static_cast<const Dtype *>(tensor_2);

        size_t i;
        int match{1};
        // every thread gets their own copy of match and after all threads are done and is called on all the values of match to assert equivalence
#pragma omp parallel for default(none) shared(t1, t2, total_len) private(i) reduction(&:match)
        for (i = 0; i < total_len; ++i) {
            if (t1[i] != t2[i]) match = 0;
        }

        *result = match;
    }

    void StandardMath::equals(
        const void *tensor_1,
        const void *tensor_2,
        int * result,
        const size_t *tensor_shape,
        const size_t *tensor_stride,
        const Dtype dtype) {

        const size_t total_len{calculate_total_elements(tensor_shape, tensor_stride)};
        switch (dtype) {
            case FLOAT64: {
                const auto casted_t1 = static_cast<const double *>(tensor_1);
                const auto casted_t2 = static_cast<const double *>(tensor_2);
                eq_kernel<double>(casted_t1, casted_t2, result, total_len);
                return;
            }
            case FLOAT32: {
                const auto casted_t1 = static_cast<const float *>(tensor_1);
                const auto casted_t2 = static_cast<const float *>(tensor_2);
                eq_kernel<float>(casted_t1, casted_t2, result, total_len);
                return;
            }
            case INT8: {
                const auto casted_t1 = static_cast<const int8_t *>(tensor_1);
                const auto casted_t2 = static_cast<const int8_t *>(tensor_2);
                eq_kernel<int8_t>(casted_t1, casted_t2, result, total_len);
                return;
            }
            case INT16: {
                const auto casted_t1 = static_cast<const int16_t *>(tensor_1);
                const auto casted_t2 = static_cast<const int16_t *>(tensor_2);
                eq_kernel<int16_t>(casted_t1, casted_t2, result, total_len);
                return;
            }
            case INT32: {
                const auto casted_t1 = static_cast<const int32_t *>(tensor_1);
                const auto casted_t2 = static_cast<const int32_t *>(tensor_2);
                eq_kernel<int32_t>(casted_t1, casted_t2, result, total_len);
                return;
            }
            case INT64: {
                const auto casted_t1 = static_cast<const int64_t *>(tensor_1);
                const auto casted_t2 = static_cast<const int64_t *>(tensor_2);
                eq_kernel<int64_t>(casted_t1, casted_t2, result, total_len);
                return;
            }
            default:
                throw std::runtime_error("cannot calculate gemv on invalid type");
        }
    }

}
