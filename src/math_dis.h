//
// Created by sriram on 12/19/24.
//

#ifndef MATHDISPATCHER_H
#define MATHDISPATCHER_H
#include "enums.h"
#include <memory>

namespace cobraml::core {
    class Math {
    public:
        virtual ~Math() = default;

        virtual bool equals(
            const void * tensor_1,
            void * tensor_2,
            const size_t * tensor_shape,
            const size_t * tensor_stride,
            Dtype dtype) = 0;

        /**
         * permutes the data in tensor to fit the shape derived from the mask
         * For example original_shape = [5, 10] permute_mask = [1, 0]
         * this means dest_shape = [10, 50 so tensor[i][j] = dest[j][i]
         * @param tensor the original data
         * @param dest the permuted tensor being modified
         * @param shape_len how many dimensions are present in the shape
         * @param original_shape the shape of the tensor
         * @param permute_mask the new ordering of the shape for the permuted tensor
         * @param original_stride the stride of the original tensor
         * @param dest_stride the stride of the permuted tensor
         * @param dtype the datatype
         */
        virtual void permute(
            const void * tensor,
            void * dest,
            size_t shape_len,
            const size_t *original_shape,
            const size_t * permute_mask,
            const size_t * original_stride,
            const size_t * dest_stride,
            Dtype dtype) = 0;

        /**
         * generalized matrix vector multiplication
         * y=α×A×x+β×y
         *
         * @param matrix A
         * @param vector x
         * @param dest y
         * @param alpha α
         * @param beta β
         * @param rows length of y
         * @param columns second dimension of A
         * @param row_stride how many columns to jump to get to the
         * next row of A (maybe more than variable columns if padding is present)
         * @param dtype the datatype
         */
        virtual void gemv(const void *matrix,
                          const void *vector,
                          void *dest,
                          const void *alpha,
                          const void *beta,
                          size_t rows,
                          size_t columns,
                          size_t row_stride,
                          Dtype dtype) = 0;

        virtual void hadamard_product(const void *tensor_one,
                                      const void *tensor_two,
                                      void *tensor_dest,
                                      const size_t *shape,
                                      size_t shape_len,
                                      const size_t *stride_one,
                                      const size_t *stride_two,
                                      size_t dest_row_stride,
                                      Dtype dtype) = 0;

        // virtual void hadamard_division(const void *matrix_one,
        //                                const void *matrix_two,
        //                                void *matrix_dest,
        //                                size_t rows,
        //                                size_t columns,
        //                                size_t row_stride,
        //                                Dtype dtype);
        //
        virtual void element_wise_add(const void *tensor_one,
                                      const void *tensor_two,
                                      void *tensor_dest,
                                      const size_t *shape,
                                      size_t shape_len,
                                      const size_t *stride_one,
                                      const size_t *stride_two,
                                      size_t dest_row_stride,
                                      Dtype dtype) = 0;
        //
        // virtual void element_wise_sub(const void *tensor_one,
        //                               const void *tensor_two,
        //                               void *tensor_dest,
        //                               size_t rows,
        //                               size_t columns,
        //                               size_t row_stride,
        //                               Dtype dtype) = 0;
        //
        // virtual void element_wise_power(const void *tensor,
        //                                 const void *exponent_tensor,
        //                                 void *tensor_dest,
        //                                 size_t rows,
        //                                 size_t columns,
        //                                 size_t row_stride,
        //                                 Dtype dtype) = 0;

        // virtual void element_wise_natural_logarithm(const void *matrix_one,
        //                                             const void *matrix_two,
        //                                             void *matrix_dest,
        //                                             size_t rows,
        //                                             size_t columns,
        //                                             size_t row_stride,
        //                                             Dtype dtype);
    };

    extern std::array<std::unique_ptr<Math>, 4> global_math_kernels;

    Math *get_math_kernels(Device device);
}

#endif //MATHDISPATCHER_H
