//
// Created by sriram on 4/18/25.
//

#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H
#include <cuda_runtime.h>
#include <iostream>


namespace cobraml::core {

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



#endif //CUDA_HELPERS_H
