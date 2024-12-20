//
// Created by sriram on 12/15/24.
//

#include "standard_allocator.h"
#include <cstdlib>
#include <cstring>
#include <iostream>


namespace cobraml::core {
    void * StandardAllocator::malloc(const std::size_t bytes) {
        return std::malloc(bytes);
    }

    void * StandardAllocator::calloc(const std::size_t bytes) {
        return std::calloc(bytes, 1);
    }

    void StandardAllocator::mem_copy(void *dest, void *source, std::size_t bytes, bool overlap) {
        if (overlap) {
            std::memmove(dest, source, bytes);
            return;
        }

        std::memcpy(dest, source, bytes);
    }

    void StandardAllocator::free(void *ptr) {
        std::free(ptr);
    }
}
