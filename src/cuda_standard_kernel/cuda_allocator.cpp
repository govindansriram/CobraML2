//
// Created by sriram on 4/18/25.
//

#include "cuda_allocator.h"
#include "cuda_helpers.h"
#include <cmath>
#include <cstring>

namespace cobraml::core {

#define GPU_ALIGNMENT 32
#define STREAM_COUNT 6

    /**
     * gets the next available event, creates a new event if no events can be reused
     * @return
     */
    cudaEvent_t EventPool::get_available_event() {
        cudaEvent_t evnt;
        if (q.empty()) {
            CUDA_CHECK(cudaEventCreate(&evnt));
            return evnt;
        }

        evnt = q.front();
        q.pop();

        return evnt;
    }

    /**
     * stores a used event back into the q for reuse
     * @param event
     */
    void EventPool::register_event(cudaEvent_t event) {
        q.push(event);
    }

    /**
     * releases the event pool
     */
    EventPool::~EventPool() {
        while (!q.empty()) {
            CUDA_CHECK(cudaEventDestroy(q.front()));
            q.pop();
        }
    }

    EventPool &get_event_pool() {
        static EventPool e_pool;
        return e_pool;
    }

    /**
     * Constructs a GpuAllocatorContext which is tied to an event
     * @param event the event tied to this context
     * @param stream the stream handling the operation
     */
    CudaContext::CudaContext(
        cudaEvent_t event,
        cudaStream_t stream): event(event), stream(stream){}

    /**
     * pauses the thread till event has completed
     */
    void CudaContext::flush() {
        cudaEventSynchronize(event);
    }

    /**
     * releases the event pool
     */
    CudaContext::~CudaContext() {
        get_event_pool().register_event(this->event);
        this->event = nullptr;
    }

    bool CudaContext::is_compute() {
        return stream == nullptr;
    }

    /**
     * creates a stream pool, which is an object that holds n unique streams allowing parallelized
     * memory operations
     */
    StreamPool::StreamPool(): streams(std::vector<cudaStream_t>(STREAM_COUNT)){
        for (size_t i{0}; i < STREAM_COUNT; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    /**
     * releases the stream pool
     */
    StreamPool::~StreamPool() {
        for (size_t i{0}; i < STREAM_COUNT; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    /**
     * @return the next available stream
     */
    cudaStream_t &StreamPool::get_stream() {
        cudaStream_t &current{streams[available_stream]};
        // TODO make this use the least accessed stream rather then the next one
        available_stream = static_cast<int8_t>((available_stream + 1) % STREAM_COUNT); // rotates to the next stream
        return current;
    }

    cudaStream_t &get_stream() {
        static StreamPool pool;
        return pool.get_stream();
    }

    /**
     * asynchronously allocates aligned data
     * @param dest a pointer to a pointer, the address being pointed will be the start of the GPU data
     * @param shape the shape of the data needing to be allocated
     * @param dtype_size size of the dtype in bytes
     * @return the stream in charge of the operation, and the padded column size
     */
    static std::pair<cudaStream_t&, size_t> internal_malloc(
        void **dest,
        const std::vector<size_t>& shape,
        const size_t dtype_size) {

        cudaStream_t& str = get_stream();
        const size_t aligned_size{compute_aligned_size(
            calculate_total_rows(shape),
            shape[shape.size() - 1],
            dtype_size,
            GPU_ALIGNMENT)};

        CUDA_CHECK(cudaMallocAsync(dest, aligned_size, str));
        return {str, aligned_size};
    }

    // TODO https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu

    std::pair<std::unique_ptr<BufferContext>, size_t> CudaAllocator::malloc(
        void **dest,
        const std::vector<size_t> &shape,
        const Dtype dtype) {

        const size_t dtype_size(dtype_to_bytes(dtype));
        auto [stream, sz]{internal_malloc(dest, shape, dtype_size)};
        cudaEvent_t evt{get_event_pool().get_available_event()};
        CUDA_CHECK(cudaEventRecord(evt, stream));
        return std::make_pair(
            std::make_unique<CudaContext>(evt, stream),
            sz
        );
    }

    std::pair<std::unique_ptr<BufferContext>, size_t> CudaAllocator::calloc(
        void **dest,
        const std::vector<size_t> &shape,
        Dtype const dtype) {

        const size_t dtype_size(dtype_to_bytes(dtype));
        auto [stream, sz]{internal_malloc(dest, shape, dtype_size)};
        CUDA_CHECK(cudaMemsetAsync(*dest, 0, sz, stream));
        cudaEvent_t evt{get_event_pool().get_available_event()};
        CUDA_CHECK(cudaEventRecord(evt, stream));

        return std::make_pair(
            std::make_unique<CudaContext>(evt, stream),
            sz
        );
    }

    cudaMemcpyKind interpret_direction(const MemoryDirection direction) {
        switch (direction) {
            case DEVICE_TO_DEVICE:
                return cudaMemcpyDeviceToDevice;
            case DEVICE_TO_HOST:
                return cudaMemcpyDeviceToHost;
            case HOST_TO_DEVICE:
                return cudaMemcpyHostToDevice;
            default:
                throw std::runtime_error("invalid cuda memory direction provided");
        }
    }

    std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext>> CudaAllocator::mem_copy(
        void *dest,
        const void *source,
        const std::size_t bytes,
        const MemoryDirection direction,
        BufferContext * dest_ctx,
        BufferContext * source_ctx) {

        // ensure all previous events completed to avoid concurrent modification
        dest_ctx->flush();
        source_ctx->flush();

        // we create 2 events to avoid returning duplicate events to the pool after a destructor call
        cudaStream_t &stream{get_stream()};
        cudaEvent_t evt1 = get_event_pool().get_available_event();
        cudaEvent_t evt2 = get_event_pool().get_available_event();
        CUDA_CHECK(cudaMemcpyAsync(dest, source, bytes, interpret_direction(direction), stream));
        CUDA_CHECK(cudaEventRecord(evt1, stream));
        CUDA_CHECK(cudaEventRecord(evt2, stream));

        return {
            std::make_unique<CudaContext>(evt1, stream),
            std::make_unique<CudaContext>(evt2, stream),
        };
    }

    std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext>> CudaAllocator::strided_mem_copy(
        void *dest,
        const void *source,
        const size_t bytes,
        const MemoryDirection direction,
        BufferContext *dest_ctx,
        BufferContext *source_ctx,
        const size_t column_count,
        const size_t padding_dest,
        const size_t padding_source) {

        if (bytes % column_count != 0)
            throw std::runtime_error("the amount of bytes being copied must be divisible by column count");

        size_t scale{bytes / column_count};

        dest_ctx->flush();
        source_ctx->flush();
        cudaStream_t &stream{get_stream()};
        cudaEvent_t evt1{get_event_pool().get_available_event()};
        cudaEvent_t evt2{get_event_pool().get_available_event()};

        if (padding_dest == padding_source) {
            scale *= padding_dest + column_count;
            CUDA_CHECK(cudaMemcpyAsync(dest, source, scale, interpret_direction(direction), stream));
            CUDA_CHECK(cudaEventRecord(evt1, stream));
            CUDA_CHECK(cudaEventRecord(evt2, stream));
            return {
                std::make_unique<CudaContext>(evt1, stream),
                std::make_unique<CudaContext>(evt2, stream)
            };
        }

        for (size_t i{0}; i < scale; ++i) {
            const auto c_dest{static_cast<char *>(dest) + i * (padding_dest + column_count)};
            const auto s_dest{static_cast<const char *>(source) + i * (padding_source + column_count)};
            CUDA_CHECK(cudaMemcpyAsync(c_dest, s_dest, column_count, interpret_direction(direction), stream));
        }
        CUDA_CHECK(cudaEventRecord(evt1, stream));
        CUDA_CHECK(cudaEventRecord(evt2, stream));

        return {std::make_unique<CudaContext>(evt1, stream), std::make_unique<CudaContext>(evt2, stream)};
    }

    void CudaAllocator::free(void *ptr, BufferContext * ctx) {
        // free should be the last operation so a new event does not need to be created
        ctx->flush();
        CUDA_CHECK(cudaFreeAsync(ptr, get_stream()));
    }
}
