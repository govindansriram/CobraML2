//
// Created by sriram on 11/24/24.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <cstddef>
#include <memory>
#include "enums.h"

namespace cobraml::core {

    class Allocator {
    public:
        virtual ~Allocator() = default;

        /**
         * allocates data for a barray, matrix, or tensor. The amount of bytes allocated will be accordance to the
         * cache alignment of the used architecture. This can only be more than requested never less.
         *
         * @param dest a ptr to the ptr which will contain the data
         * @param total_rows the total amount of rows in the data structure, to find the total amount of bytes allocated
         * use total_rows * return
         * @param total_columns the amount of columns requested in the data structure
         * @param dtype_size size of the dtype in bytes, i.e. float is 4
         * @return the amount of data allocated per row of the data structure
         */
        virtual size_t malloc(void ** dest, size_t total_rows, size_t total_columns, size_t dtype_size) = 0;

        /**
         *  allocates data for a barray, matrix, or tensor, with default values set to 0. The amount of bytes allocated
         *  will be accordance to the cache alignment of the used architecture. This can only be more than requested
         *  never less.
         * @param dest a ptr to the ptr which will contain the data
        * @param total_rows the total amount of rows in the data structure, to find the total amount of bytes allocated
         * use total_rows * return
         * @param total_columns the amount of columns requested in the data structure
         * @param dtype_size size of the dtype in bytes, i.e. float is 4
         * @return the amount of data allocated per row of the data structure
         */
        virtual size_t calloc(void ** dest, size_t total_rows, size_t total_columns, size_t dtype_size) = 0;
        virtual void mem_copy(void *dest, const void *source, std::size_t bytes) = 0;
        virtual void free(void *ptr) = 0;
    };

    extern std::array<std::unique_ptr<Allocator>, 3> global_allocators;
    Allocator * get_allocator(Device device);

    // class Buffer {
    //     void * p_buffer = nullptr;
    //     Allocator * p_allocator;
    //     Device device;
    //
    // public:
    //     Buffer() = delete;
    //     explicit Buffer(size_t bytes, Device device);
    //     ~Buffer();
    //     [[nodiscard]] void * get_p_buffer() const;
    //     Buffer(Buffer&) = delete;
    //     Buffer& operator=(Buffer&) = delete;
    //
    //     /**
    //      * overwrite a segment of the buffer with new data
    //      * @param source the new data
    //      * @param byte_count how many bytes to replace
    //      * @param offset the starting position to start overwriting in the original buffer
    //      */
    //     void overwrite(const void * source, size_t byte_count, size_t offset = 0) const;
    // };
}

#endif //ALLOCATOR_H
