//
// Created by sriram on 4/18/25.
//

#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <vector>
#include <cuda_runtime.h>
#include "../allocator.h"

namespace cobraml::core {
    class StreamPool {
        std::vector<cudaStream_t> streams;
        size_t available_stream{0}; // track workload better in the future

    public:
        StreamPool();

        ~StreamPool();

        cudaStream_t get_stream();
    };

    class CudaAllocator final : public Allocator {

    public:
        size_t malloc(void **dest, const std::vector<size_t> &shape, Dtype dtype) override;
        size_t calloc(void **dest, const std::vector<size_t> &shape, Dtype dtype) override;

        void mem_copy(
            void *dest,
            const void *source,
            std::size_t bytes,
            MemoryDirection direction) override;

        void strided_mem_copy(
            void *dest,
            const void *source,
            size_t bytes,
            MemoryDirection direction,
            size_t column_count,
            size_t padding_dest,
            size_t padding_source) override;

        void free(void *ptr) override;
    };

    cudaStream_t get_stream();
}


#endif //CUDA_ALLOCATOR_H
