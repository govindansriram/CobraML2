//
// Created by sriram on 12/15/24.
//

#include "standard_allocator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>


namespace cobraml::core {

#define ALIGNMENT 32
#define MIN_LENGTH 256 // 64  * 4 (ensures there is a 4 element padding for 64 bit systems)

    static size_t compute_aligned_size(size_t const bytes) {

        size_t const bits = bytes * 8;

        size_t const remainder = bits % MIN_LENGTH;
        size_t quotient = bits / MIN_LENGTH;

        quotient += remainder > 0 ? 1 : 0;

        // if (remainder == bytes)
        //     return MIN_LENGTH;
        //
        // if (remainder == 0)
        //     return bytes;

        // std::cout << bits << " " << quotient * (MIN_LENGTH / 8) << std::endl;

        return quotient * (MIN_LENGTH / 8);
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
