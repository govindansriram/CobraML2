//
// Created by sriram on 1/25/25.
//

#ifndef BARRAY_H
#define BARRAY_H
#include <memory>
#include <vector>
#include "enums.h"

namespace cobraml::core {
    class Array {

    protected:
        struct ArrayImpl;
        std::unique_ptr<ArrayImpl> impl;

        void increment_offset(unsigned long inc);
        void set_length(unsigned long len);

        /**
         * get the raw pointer backing an array, do not free this pointer
         *
         * @return a raw void pointer to the buffer data
         */
        [[nodiscard]] void *get_raw_buffer() const;

        /**
         * replace a segment of the array buffer with a different buffer
         * @param source the replacement data
         * @param items how many items we are extracting
         */
        void replace_segment(const void * source, size_t items) const;

        template<typename T>
        void copy_vector(const std::vector<T> &vec) {

            constexpr Dtype dtype{get_dtype_from_type<T>::type};
            is_invalid(dtype);

            if (const Dtype current{get_dtype()}; dtype != current) {
                throw std::runtime_error("vector dtype does not match array dtype");
            }

            if (vec.size() != len())
                throw std::runtime_error("cannot set array with vector of different size");

            replace_segment(vec.data(), vec.size());
        }

        // Mathematical operations

        /**
         * Generalized Matrix Vector Multiplication.
         * Performs y=αAx+βy
         *
         * @param matrix A
         * @param vector x
         * @param rows
         * @param columns
         * @param alpha α
         * @param beta β
         */
        void gemv(
            const Array &matrix,
            const Array &vector,
            size_t rows,
            size_t columns,
            const void * alpha,
            const void * beta);

    public:
        Array(size_t total_items, Device device, Dtype dtype);
        virtual ~Array();
        Array();
        Array(const Array &other);
        Array& operator=(const Array& other);
        [[nodiscard]] size_t len() const;
        [[nodiscard]] virtual Array deep_copy() const;
        virtual void print(bool show_description) const;

        /**
         * @return the dtype of the matrix
         */
        [[nodiscard]] Dtype get_dtype() const;

        /**
        * @return the Device of the matrix
        */
        [[nodiscard]] Device get_device() const;

        /**
         * provides access to the underlying buffer
         * @tparam T the type that the ptr should be cast too, it must match the Dtype
         * @return the raw ptr buffer
         */
        template<typename T>
        friend const T *get_buffer(const Array &arr);

        template<typename T>
        friend Array from_vector(std::vector<T> vec, Device device, Dtype dtype);

        Array operator[](size_t index) const;

        template<typename T>
        T item() const {
            const Dtype current{this->get_dtype()};
            if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
                throw std::runtime_error(
                    "provided buffer type does not match array type: " + dtype_to_string(current));
            }

            if (this->len() != 1) {
                throw std::out_of_range("array can only return item if it contains a single element");
            }

            const T * buff = static_cast<T *>(get_raw_buffer());
            return *buff;
        }

        template<typename T>
        void set_item(T value) const {
            const Dtype current{this->get_dtype()};
            if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
                throw std::runtime_error(
                    "provided buffer type does not match array type: " + dtype_to_string(current));
            }

            if (this->len() != 1) {
                throw std::out_of_range("array can only set 1 item at a time");
            }

            replace_segment(&value, 1);
        }
    };

    template<typename T>
    const T *get_buffer(const Array &arr) {
        const Dtype current{arr.get_dtype()};
        if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
            throw std::runtime_error(
                "provided buffer type does not match array type: " + dtype_to_string(current));
        }

        return static_cast<T *>(arr.get_raw_buffer());
    }

    template<typename T>
    Array from_vector(std::vector<T> vec, Device const device, Dtype const dtype) {
        Array ret(vec.size(), device, dtype);
        ret.copy_vector(vec);
        return ret;
    }
}

#endif //BARRAY_H
