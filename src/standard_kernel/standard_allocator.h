//
// Created by sriram on 12/15/24.
//

#ifndef STANDARD_ALLOCATOR_H
#define STANDARD_ALLOCATOR_H

#include "../allocator.h"

namespace cobraml::core {
    class StandardAllocator final : public Allocator {
    public:
         size_t malloc(
            void **dest,
            const std::vector<size_t>& shape,
            Dtype dtype) override;

         size_t calloc(
            void **dest,
            const std::vector<size_t>& shape,
            Dtype dtype) override;

        void mem_copy(
            void *dest,
            const void *source,
            std::size_t bytes,
            MemoryDirection direction) override;

        void strided_mem_copy(
            void *dest,
            const void *source,
            size_t bytes,
            MemoryDirection direction,
            size_t column_count,
            size_t padding_dest,
            size_t padding_source) override;

        void free(void *ptr) override;

        ~StandardAllocator() override = default;
    };
}

#endif //STANDARD_ALLOCATOR_H
