//
// Created by sriram on 4/18/25.
//

#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <vector>
#include <cuda_runtime.h>
#include <queue>

#include "../allocator.h"

namespace cobraml::core {
    class StreamPool {
        std::vector<cudaStream_t> streams;
        int8_t available_stream{0}; // track workload better in the future

    public:
        StreamPool();

        ~StreamPool();

        cudaStream_t &get_stream();
    };

    class EventPool {
        std::queue<cudaEvent_t> q{};

        ~EventPool();

    public:
        cudaEvent_t get_available_event();

        void register_event(cudaEvent_t);
    };

    struct CudaContext final : BufferContext {
        CudaContext(cudaEvent_t event, cudaStream_t stream);

        ~CudaContext() override;

        void flush() override;

        bool is_compute() override;

    private:
        cudaEvent_t event;
        cudaStream_t stream{nullptr};
    };

    class CudaAllocator final : public Allocator {
        std::pair<std::unique_ptr<BufferContext>, size_t> malloc(void **dest, const std::vector<size_t> &shape,
                                                                 Dtype dtype) override;

        std::pair<std::unique_ptr<BufferContext>, size_t> calloc(void **dest, const std::vector<size_t> &shape,
                                                                 Dtype dtype) override;

        std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext> > mem_copy(
            void *dest,
            const void *source,
            std::size_t bytes,
            MemoryDirection direction,
            BufferContext *dest_ctx,
            BufferContext *source_ctx) override;

        std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext> > strided_mem_copy(
            void *dest,
            const void *source,
            size_t bytes,
            MemoryDirection direction,
            BufferContext *dest_ctx,
            BufferContext *source_ctx,
            size_t column_count,
            size_t padding_dest,
            size_t padding_source) override;

        void free(void *ptr, BufferContext *ctx) override;
    };

    cudaStream_t &get_stream();

    EventPool &get_event_pool();
}


#endif //CUDA_ALLOCATOR_H
