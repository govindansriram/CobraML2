//
// Created by sriram on 4/26/25.
//

#include "cuda_math.h"


namespace cobraml::core {

void CudaMath::element_wise_add([[maybe_unused]] const void *tensor_one,
                                 [[maybe_unused]] const void *tensor_two,
                                 [[maybe_unused]] void *tensor_dest,
                                 [[maybe_unused]] const size_t *shape,
                                 [[maybe_unused]] size_t shape_len,
                                 [[maybe_unused]] const size_t *stride_one,
                                 [[maybe_unused]] const size_t *stride_two,
                                 [[maybe_unused]] size_t dest_row_stride,
                                 [[maybe_unused]] Dtype dtype) {
    throw std::runtime_error("coming soon");
}

void CudaMath::hadamard_product([[maybe_unused]] const void *tensor_one,
                                 [[maybe_unused]] const void *tensor_two,
                                 [[maybe_unused]] void *tensor_dest,
                                 [[maybe_unused]] const size_t *shape,
                                 [[maybe_unused]] size_t shape_len,
                                 [[maybe_unused]] const size_t *stride_one,
                                 [[maybe_unused]] const size_t *stride_two,
                                 [[maybe_unused]] size_t dest_row_stride,
                                 [[maybe_unused]] Dtype dtype) {
    throw std::runtime_error("coming soon");
}

void CudaMath::permute([[maybe_unused]] const void *tensor,
                       [[maybe_unused]] void *dest,
                       [[maybe_unused]] size_t shape_len,
                       [[maybe_unused]] const size_t *original_shape,
                       [[maybe_unused]] const size_t *permute_mask,
                       [[maybe_unused]] const size_t *original_stride,
                       [[maybe_unused]] const size_t *dest_stride,
                       [[maybe_unused]] Dtype dtype) {
    throw std::runtime_error("coming soon");
}




}