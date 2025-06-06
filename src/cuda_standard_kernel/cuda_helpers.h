//
// Created by sriram on 4/18/25.
//

#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>


namespace cobraml::core {
    cublasHandle_t get_handle();

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            std::cerr << "CUDA error at " << __FILE__ << ":"     \
            << __LINE__ << " code=" << static_cast<int>(err)     \
            << " \"" << cudaGetErrorString(err) << "\"\n";       \
            std::exit(EXIT_FAILURE);                             \
        }                                                        \
    } while (0)
}

#define TILE_WIDTH 16

template<typename t1, typename t2>
unsigned int calculate_dim(t1 first, t2 second) {
    return std::ceil(static_cast<float>(first) / static_cast<float>(second));
}

#endif //CUDA_HELPERS_H
