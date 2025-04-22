//
// Created by sriram on 11/24/24.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include "context.h"
#include "enums.h"

namespace cobraml::core {

    /**
     * Computes the aligned size in bytes, for this method to work
     * all shapes must be flattened to a 2 dim tensor (matrix)
     * @param total_rows
     * @param total_columns
     * @param dtype_size
     * @param alignment_size
     * @return
     */
    inline size_t compute_aligned_size(
        size_t const total_rows,
        size_t const total_columns,
        size_t const dtype_size,
        size_t const alignment_size) {

        if (alignment_size % dtype_size) {
            // TODO: Add the alignment value into the error
            throw std::runtime_error("dtype is not a factor of the required alignment");
        }

        const auto requested{total_columns * dtype_size};

        if (requested < alignment_size)
            return alignment_size * total_rows;

        if (!(requested % alignment_size)) {
            // requested is a direct multiple of ALIGNMENT
            return requested * total_rows;
        }

        // round to the closest multiple
        size_t const multiplier{static_cast<size_t>(
            std::ceil(static_cast<float>(requested) / static_cast<float>(alignment_size)))};

        return multiplier * alignment_size * total_rows;
    }

    inline size_t calculate_total_rows(std::vector<size_t> const &shape) {
        if (shape.empty()) throw std::runtime_error("shape cannot be empty");
        if (shape[0] == 0) throw std::runtime_error("dimensions in shape cannot be zero");
        size_t rows{1};
        for (size_t i{1}; i < shape.size(); ++i){
            if (shape[i] == 0) throw std::runtime_error("dimensions in shape cannot be zero");
            rows *= shape[i];
        }
        return rows;
    }


    class Allocator {
    public:
        virtual ~Allocator() = default;

        /**
         * allocates a memory buffer for a brarray. The amount of bytes allocated may be padded for better
         * cache alignment with respect to the architecture. This can only be more than requested, never less.
         *
         * @param dest a ptr to the ptr which will contain the data
         * @param shape these are the dimensions of the brarray, internally we flatten this to one dimension to compute
         * the requested memory amount
         * @param dtype the datatype of the buffer, lets us determine how many bytes are needed per element
         * @return a unique ptr containing the context for the buffer, and the total bytes allocated. To determine
         * the total elements, divide by dtype size. To determine the stride flatten the shape to 2 dimensions
         * divide total bytes by the value of the first dimension then divide by dtype size
         */
        virtual std::pair<std::unique_ptr<BufferContext>, size_t> malloc(
            void **dest,
            const std::vector<size_t>& shape,
            Dtype dtype) = 0;

        /**
         * allocates a memory buffer for a brarray and sets all values to zero. The amount of bytes allocated may be
         * padded for better cache alignment with respect to the architecture. This can only be more than requested,
         * never less.
         *
         * @param dest a ptr to the ptr which will contain the data
         * @param shape these are the dimensions of the brarray, internally we flatten this to one dimension to compute
         * the requested memory amount
         * @param dtype the datatype of the buffer, lets us determine how many bytes are needed per element
         * @return a unique ptr containing the context for the buffer, and the total bytes allocated. To determine
         * the total elements, divide by dtype size. To determine the stride flatten the shape to 2 dimensions
         * divide total bytes by the value of the first dimension then divide by dtype size
         */
        virtual std::pair<std::unique_ptr<BufferContext>, size_t> calloc(
            void **dest,
            const std::vector<size_t>& shape,
            Dtype dtype) = 0;

        virtual std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext>> mem_copy(
            void *dest,
            const void *source,
            std::size_t bytes,
            MemoryDirection direction,
            BufferContext * dest_ctx,
            BufferContext * source_ctx) = 0;

        virtual std::pair<std::unique_ptr<BufferContext>, std::unique_ptr<BufferContext>> strided_mem_copy(
            void *dest,
            const void *source,
            size_t bytes,
            MemoryDirection direction,
            BufferContext * dest_ctx,
            BufferContext * source_ctx,
            size_t column_count,
            size_t padding_dest,
            size_t padding_source);

        virtual void free(
            void *ptr,
            BufferContext * ctx) = 0;
    };

    extern std::array<std::unique_ptr<Allocator>, 4> global_allocators;
    Allocator * get_allocator(Device device);
}

#endif //ALLOCATOR_H
