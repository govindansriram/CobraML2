//
// Created by sriram on 5/11/25.
//

#include "cuda_helpers.h"
#include <cublas_v2.h>

namespace cobraml::core {

    struct CublasHandler {
        cublasHandle_t handle{};

        CublasHandler() {
            cublasCreate(&handle);
        }

        ~CublasHandler() {
            cublasDestroy(handle);
        }

        CublasHandler& operator=(const CublasHandler&) = delete;
        CublasHandler(const CublasHandler&) = delete;
    };

    cublasHandle_t get_handle() {
        static CublasHandler handler;
        return handler.handle;
    }
}
