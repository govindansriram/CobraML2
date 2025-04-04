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
                                      size_t rows,
                                      size_t columns,
                                      size_t row_stride,
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
                                      size_t rows,
                                      size_t columns,
                                      size_t row_stride,
                                      Dtype dtype) = 0;

        virtual void element_wise_sub(const void *tensor_one,
                                      const void *tensor_two,
                                      void *tensor_dest,
                                      size_t rows,
                                      size_t columns,
                                      size_t row_stride,
                                      Dtype dtype) = 0;

        virtual void element_wise_power(const void *tensor,
                                        const void *exponent_tensor,
                                        void *tensor_dest,
                                        size_t rows,
                                        size_t columns,
                                        size_t row_stride,
                                        Dtype dtype) = 0;

        // virtual void element_wise_natural_logarithm(const void *matrix_one,
        //                                             const void *matrix_two,
        //                                             void *matrix_dest,
        //                                             size_t rows,
        //                                             size_t columns,
        //                                             size_t row_stride,
        //                                             Dtype dtype);
    };

    extern std::array<std::unique_ptr<Math>, 3> global_math_kernels;

    Math *get_math_kernels(Device device);
}

#endif //MATHDISPATCHER_H
