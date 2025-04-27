//
// Created by sriram on 4/26/25.
//

#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include "enums.h"
#include "../math_dis.h"


namespace cobraml::core {
    class CudaMath final : public Math {
    public:
        bool equals(
            const void *tensor_1,
            void *tensor_2,
            const size_t *tensor_shape,
            const size_t *tensor_stride,
            Dtype dtype) override;

        void permute(
            const void *tensor,
            void *dest,
            size_t shape_len,
            const size_t *original_shape,
            const size_t *permute_mask,
            const size_t *original_stride,
            const size_t *dest_stride,
            Dtype dtype) override;

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


#endif //CUDA_MATH_H
