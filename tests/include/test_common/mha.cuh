#pragma once
#include "./initialize.cuh"

namespace cobraml::test_helpers{

    template<typename DType>
    thrust::device_vector<DType> create_projection(
        int head_count, 
        int head_dim, 
        int batch_size, 
        int sequence_length,
        auto fill_fn
    ){
        int total_length{head_count * head_dim * batch_size * sequence_length};
        return create_tensor<DType>(total_length, fill_fn);
    }

}