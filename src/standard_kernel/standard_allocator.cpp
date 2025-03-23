//
// Created by sriram on 12/15/24.
//

#include "standard_allocator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>


namespace cobraml::core {

#ifdef AVX2
    #define ALIGNMENT 32
#else
    #define ALIGNMENT 8
#endif


    static size_t compute_aligned_size(size_t const total_rows, size_t const total_columns, size_t const dtype_size) {
        if (ALIGNMENT % dtype_size) {
            // TODO: Add the alignment value into the error
            throw std::runtime_error("dtype is not a factor of the required alignment");
        }

        const auto requested{total_columns * dtype_size};

        if (requested < ALIGNMENT)
            return ALIGNMENT * total_rows;

        if (!(requested % ALIGNMENT)) { // requested is a direct multiple of ALIGNMENT
            return requested * total_rows;
        }

        // round to the closest multiple
        size_t const multiplier{static_cast<size_t>(std::ceil(static_cast<float>(requested) / ALIGNMENT))};
        return multiplier * ALIGNMENT * total_rows;
    }

    size_t StandardAllocator::malloc(void ** dest, size_t const total_rows, size_t const total_columns, size_t const dtype_size) {
        size_t const size = compute_aligned_size(total_rows, total_columns, dtype_size);
        *dest = std::aligned_alloc(ALIGNMENT, size);
        return size / total_rows;
    }


    // A malloc() followed by a memset() will likely be about as fast as calloc()
    // https://stackoverflow.com/questions/2605476/calloc-v-s-malloc-and-time-efficiency
    size_t StandardAllocator::calloc(void ** dest, size_t const total_rows, size_t const total_columns, size_t const dtype_size) {
        size_t const column_length = malloc(dest, total_rows, total_columns, dtype_size);
        std::memset(*dest, 0, column_length * total_rows);
        return column_length;
    }

    void StandardAllocator::mem_copy(void *dest, const void *source, std::size_t const bytes) {
        std::memcpy(dest, source, bytes);
    }

    void StandardAllocator::free(void *ptr) {
        std::free(ptr);
    }
}
