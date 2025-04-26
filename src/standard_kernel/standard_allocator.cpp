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


    size_t StandardAllocator::malloc(
        void **dest,
        const std::vector<size_t>& shape,
        Dtype const dtype) {
        const size_t dtype_size(dtype_to_bytes(dtype));

        const size_t aligned_size{compute_aligned_size(
            calculate_total_rows(shape),
            shape[shape.size() - 1],
            dtype_size,
            CPU_ALIGNMENT)};

        *dest = std::aligned_alloc(CPU_ALIGNMENT, aligned_size);
        return aligned_size;
    }

    // A malloc() followed by a memset() will likely be about as fast as calloc()
    // https://stackoverflow.com/questions/2605476/calloc-v-s-malloc-and-time-efficiency
    size_t StandardAllocator::calloc(
        void **dest,
        const std::vector<size_t>& shape,
        Dtype const dtype) {
        const size_t aligned_size{malloc(dest, shape, dtype)};
        std::memset(*dest, 0, aligned_size);
        return aligned_size;
    }

    void StandardAllocator::mem_copy(
        void *dest,
        const void *source,
        const std::size_t bytes,
        const MemoryDirection direction) {
        (void)direction; // Mark as intentionally unused to silence warnings
        std::memcpy(dest, source, bytes);
    }

    void StandardAllocator::strided_mem_copy(
        void *dest,
        const void *source,
        const size_t bytes,
        const MemoryDirection direction,
        const size_t column_count,
        const size_t padding_dest,
        const size_t padding_source) {

        (void)direction; // Mark as intentionally unused to silence warnings

        if (bytes % column_count != 0)
            throw std::runtime_error("the amount of bytes being copied must be divisible by column count");

        size_t scale{bytes / column_count};

        if (padding_dest == padding_source) {
            scale *= padding_dest + column_count;
            std::memcpy(dest, source, scale);
            return;
        }

        for (size_t i{0}; i < scale; ++i) {
            const auto c_dest{static_cast<char *>(dest) + i * (padding_dest + column_count)};
            const auto s_dest{static_cast<const char *>(source) + i * (padding_source + column_count)};
            std::memcpy(c_dest, s_dest, column_count);
        }
    }

    void StandardAllocator::free(void *ptr) {
        std::free(ptr);
    }
}
