//
// Created by sriram on 12/15/24.
//

#include "standard_allocator.h"
#include <cstdlib>
#include <cstring>
#include <cmath>


namespace cobraml::core {
#ifdef AVX2
#define CPU_ALIGNMENT 32
#else
    #define CPU_ALIGNMENT 8
#endif

    struct DummyContext final : BufferContext {
        void flush() override {};
        bool is_compute() override {return false;};
    };


    std::pair<std::unique_ptr<BufferContext>, size_t> StandardAllocator::malloc(
        void **dest,
        const std::vector<size_t>& shape,
        Dtype const dtype) {
        auto ctx{std::make_unique<DummyContext>()};

        const size_t dtype_size(dtype_to_bytes(dtype));

        const size_t aligned_size{compute_aligned_size(
            calculate_total_rows(shape),
            shape[shape.size() - 1],
            dtype_size,
            CPU_ALIGNMENT)};

        *dest = std::aligned_alloc(CPU_ALIGNMENT, aligned_size);
        return std::make_pair(std::move(ctx), aligned_size);
    }

    // A malloc() followed by a memset() will likely be about as fast as calloc()
    // https://stackoverflow.com/questions/2605476/calloc-v-s-malloc-and-time-efficiency
    std::pair<std::unique_ptr<BufferContext>, size_t> StandardAllocator::calloc(
        void **dest,
        const std::vector<size_t>& shape,
        Dtype const dtype) {
        auto ret{malloc(dest, shape, dtype)};
        std::memset(*dest, 0, ret.second);
        return ret;
    }

    std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext>> StandardAllocator::mem_copy(
        void *dest,
        const void *source,
        const std::size_t bytes,
        MemoryDirection direction,
        BufferContext *dest_ctx,
        BufferContext *source_ctx) {

        // ensure no other operations are happening
        dest_ctx->flush();
        source_ctx->flush();

        std::memcpy(dest, source, bytes);
        return {
            std::make_unique<DummyContext>(), std::make_unique<DummyContext>()
        };
    }

    std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext>> StandardAllocator::strided_mem_copy(
        void *dest,
        const void *source,
        const size_t bytes,
        MemoryDirection direction,
        BufferContext *dest_ctx,
        BufferContext *source_ctx,
        const size_t column_count,
        const size_t padding_dest,
        const size_t padding_source) {

        dest_ctx->flush();
        source_ctx->flush();

        if (bytes % column_count != 0)
            throw std::runtime_error("the amount of bytes being copied must be divisible by column count");

        size_t scale{bytes / column_count};

        if (padding_dest == padding_source) {
            scale *= padding_dest + column_count;
            std::memcpy(dest, source, scale);
            return {std::make_unique<DummyContext>(), std::make_unique<DummyContext>()};
        }

        for (size_t i{0}; i < scale; ++i) {
            const auto c_dest{static_cast<char *>(dest) + i * (padding_dest + column_count)};
            const auto s_dest{static_cast<const char *>(source) + i * (padding_source + column_count)};
            std::memcpy(c_dest, s_dest, column_count);
        }

        return {std::make_unique<DummyContext>(), std::make_unique<DummyContext>()};
    }

    void StandardAllocator::free(void *ptr, BufferContext * ctx) {
        ctx->flush(); // ensure the data is no longer being used by any operation
        std::free(ptr);
    }
}
