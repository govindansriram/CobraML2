//
// Created by sriram on 12/15/24.
//

#include "standard_allocator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>


namespace cobraml::core {

#define ALIGNMENT 64

#ifdef AVX2
    #define ALIGNMENT 32
#endif


    static size_t compute_aligned_size(size_t const total_rows, size_t const total_columns, size_t const dtype_size) {
        if (!(ALIGNMENT % dtype_size)) {
            // TODO: Add the alignment value into the error
            throw std::runtime_error("dtype is not a factor of the required alignment");
        }

        const auto requested{total_columns * dtype_size};

        if (requested < ALIGNMENT)
            return ALIGNMENT * total_rows;

        if (!(requested % ALIGNMENT)) {
            return requested * total_rows;
        }

        // round to the closest multiple
        size_t const multiplier{static_cast<size_t>(std::ceil(requested / ALIGNMENT))};
        return multiplier * requested * total_rows;
    }

    void * StandardAllocator::malloc(std::size_t const bytes) {
        // std::cout << bytes << " bytes" << std::endl;
        // std::cout << compute_aligned_size(bytes) << " bytes" << std::endl;
        return std::aligned_alloc(ALIGNMENT, compute_aligned_size(bytes));
    }


    // A malloc() followed by a memset() will likely be about as fast as calloc()
    // https://stackoverflow.com/questions/2605476/calloc-v-s-malloc-and-time-efficiency
    void * StandardAllocator::calloc(const std::size_t bytes) {
        void * ptr = malloc(bytes);
        std::memset(ptr, 0, compute_aligned_size(bytes));
        return ptr;
    }

    void StandardAllocator::mem_copy(void *dest, const void *source, std::size_t const bytes) {
        std::memcpy(dest, source, bytes);
    }

    void StandardAllocator::free(void *ptr) {
        std::free(ptr);
    }
}
