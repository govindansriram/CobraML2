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
    cudaStream_t StreamPool::get_stream() {
        cudaStream_t current{streams[available_stream]};
        // TODO make this use the least accessed stream rather then the next one
        available_stream = (available_stream + 1) % STREAM_COUNT; // rotates to the next stream
        return current;
    }

    cudaStream_t get_stream() {
        static StreamPool pool;
        return pool.get_stream();
    }

    // TODO https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu

    size_t CudaAllocator::malloc(
        void **dest,
        const std::vector<size_t> &shape,
        const Dtype dtype) {

        const size_t dtype_size(dtype_to_bytes(dtype));
        const size_t aligned_size{compute_aligned_size(
            calculate_total_rows(shape),
            shape[shape.size() - 1],
            dtype_size,
            GPU_ALIGNMENT)};

        CUDA_CHECK(cudaMalloc(dest, aligned_size));
        return aligned_size;
    }

    size_t CudaAllocator::calloc(
        void **dest,
        const std::vector<size_t> &shape,
        Dtype const dtype) {

        size_t const aligned_size{malloc(dest, shape, dtype)};
        CUDA_CHECK(cudaMemset(*dest, 0, aligned_size));
        return aligned_size;
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

    void CudaAllocator::mem_copy(
        void *dest,
        const void *source,
        const std::size_t bytes,
        const MemoryDirection direction) {
        CUDA_CHECK(cudaMemcpy(dest, source, bytes, interpret_direction(direction)));
    }

    void CudaAllocator::strided_mem_copy(
        void *dest,
        const void *source,
        const size_t bytes,
        const MemoryDirection direction,
        const size_t column_count,
        const size_t padding_dest,
        const size_t padding_source) {

        if (bytes % column_count != 0)
            throw std::runtime_error("the amount of bytes being copied must be divisible by column count");

        size_t scale{bytes / column_count};

        if (padding_dest == padding_source) {
            scale *= padding_dest + column_count;
            mem_copy(dest, source, scale, direction);
            return;
        }

        for (size_t i{0}; i < scale; ++i) {
            cudaStream_t stream{get_stream()};
            const auto c_dest{static_cast<char *>(dest) + i * (padding_dest + column_count)};
            const auto s_dest{static_cast<const char *>(source) + i * (padding_source + column_count)};
            CUDA_CHECK(cudaMemcpyAsync(c_dest, s_dest, column_count, interpret_direction(direction), stream));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void CudaAllocator::free(void *ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}
