//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_ALLOCATOR_H
#define STANDARD_ALLOCATOR_H

#include "../allocator.h"

namespace cobraml::core {
    class StandardAllocator final : public Allocator {
    public:
        size_t malloc(void ** dest, size_t total_rows, size_t total_columns, size_t dtype_size) override;
        size_t calloc(void ** dest, size_t total_rows, size_t total_columns, size_t dtype_size) override;
        void mem_copy(void *dest, const void *source, std::size_t bytes) override;
        void free(void *ptr) override;
        ~StandardAllocator() override = default;
    };
}

#endif //STANDARD_ALLOCATOR_H