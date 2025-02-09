//
// Created by sriram on 1/25/25.
//

#include "barray.h"

#include <iostream>
#include <iomanip>
#include "math_dis.h"
#include "allocator.h"

namespace cobraml::core {
    struct Array::ArrayImpl {
        size_t offset = 0;
        size_t len = 0;
        Device device = CPU;
        Dtype dtype = INVALID;
        std::shared_ptr<Buffer> buffer = nullptr;
        Math *m_dispatcher = nullptr;

        ArrayImpl(Device const device, Dtype const dtype, size_t const total_items): len(total_items),
            device(device),
            dtype(dtype),
            buffer(std::make_shared<Buffer>(total_items * dtype_to_bytes(dtype), device)),
            m_dispatcher(get_math_kernels(device)) {
            if (total_items == 0) {
                throw std::runtime_error("cannot initialize a barray with size 0 items");
            }
        }

        [[nodiscard]] void *get_raw_buffer() const {
            return static_cast<char *>(buffer->get_p_buffer()) + offset;
        }

        ArrayImpl() = default;

        ArrayImpl(const ArrayImpl &) = default;

        ArrayImpl &operator=(const ArrayImpl &) = default;
    };

    Array::Array(size_t total_items, Device device, Dtype dtype): impl(
        std::make_unique<ArrayImpl>(device, dtype, total_items)) {
        is_invalid(dtype);
    }

    Dtype Array::get_dtype() const {
        return this->impl->dtype;
    }

    Device Array::get_device() const {
        return this->impl->device;
    }

    void *Array::get_raw_buffer() const {
        if (impl->buffer == nullptr) {
            throw std::runtime_error("data buffer is null");
        }

        return static_cast<char *>(impl->buffer->get_p_buffer()) + impl->offset;
    }

    Array::~Array() = default;

    Array::Array(): impl(std::make_unique<ArrayImpl>()) {
    }

    Array::Array(const Array &other) : impl(std::make_unique<ArrayImpl>()) {
        impl->dtype = other.impl->dtype;
        impl->offset = other.impl->offset;
        impl->device = other.impl->device;
        impl->buffer = other.impl->buffer;
        impl->len = other.impl->len;
        impl->m_dispatcher = other.impl->m_dispatcher;
    }

    Array &Array::operator=(const Array &other) {
        if (this == &other)
            return *this;

        impl->dtype = other.impl->dtype;
        impl->offset = other.impl->offset;
        impl->device = other.impl->device;
        impl->buffer = other.impl->buffer;
        impl->len = other.impl->len;
        impl->m_dispatcher = other.impl->m_dispatcher;

        return *this;
    }

    void Array::set_length(unsigned long const len) {
        if (len > impl->len) {
            throw std::runtime_error("cannot make length negative");
        }

        impl->len = len;
    }

    void Array::increment_offset(unsigned long const inc) {
        if (inc > impl->len - 1) {
            throw std::runtime_error("cannot make offset larger then avalailble data");
        }


        impl->offset += inc * dtype_to_bytes(impl->dtype);
    }

    size_t Array::len() const {
        return impl->len;
    }

    void Array::gemv(
        const Array &matrix,
        const Array &vector,
        size_t const rows,
        size_t const columns,
        const void *alpha,
        const void *beta) {
        this->impl->m_dispatcher->gemv(
            matrix.get_raw_buffer(),
            vector.get_raw_buffer(),
            this->get_raw_buffer(),
            alpha,
            beta,
            rows,
            columns,
            this->get_dtype());
    }

    void Array::replace_segment(const void *source, size_t items) const {
        impl->buffer->overwrite(source, items * dtype_to_bytes(get_dtype()), this->impl->offset);
    }

    Array Array::deep_copy() const{
        Array ret(this->impl->len, this->impl->device, this->impl->dtype);

        ret.impl->buffer->overwrite(
            this->get_raw_buffer(),
            this->impl->len * dtype_to_bytes(get_dtype()),
            0);

        return ret;
    }

    Array Array::operator[](size_t const index) const {
        Array ret = *this;

        if (index >= ret.len()) {
            throw std::out_of_range("index out of bounds");
        }

        ret.increment_offset(index);
        ret.set_length(1);

        return ret;
    }

#define PRINT_LIMIT 30

#define PRINT_BARRAY(p_arr, length, precision){\
    std::cout << "[";\
    size_t stop = length / 2;\
    size_t start2 = stop;\
    bool middle_dots{};\
    if(length > PRINT_LIMIT){\
        middle_dots=true;\
        start2 = length - 3;\
    }\
    for (size_t i = 0; i < stop; ++i) {\
        std::cout << std::fixed << std::setprecision(precision) << p_arr[i] << ", ";\
    }\
    if(middle_dots){\
        std::cout << "... ";\
    }\
    for (; start2 < length - 1; ++start2) {\
        std::cout << std::fixed << std::setprecision(precision) << p_arr[start2] << ", ";\
    }\
    std::cout << std::fixed << std::setprecision(precision) << p_arr[start2] << "]\n";\
}

    void print_description(Array const *arr) {
        std::cout << "############## Details ##############\n";
        std::cout << "Length: " << arr->len() << "\n";
        std::cout << "Device: " << device_to_string(arr->get_device()) << "\n";
        std::cout << "Dytpe: " << dtype_to_string(arr->get_dtype()) << "\n";
        std::cout << "#####################################\n";
    }

    void Array::print(bool const show_description) const{

        if (this->get_dtype() == INVALID) {
            throw std::runtime_error("cannot print barray with invalid dtype");
        }

        if (show_description)
            print_description(this);

        switch (this->impl->dtype) {
            case INT8: {
                auto const p{static_cast<int8_t *>(this->get_raw_buffer())};
                PRINT_BARRAY(p, len(), 0);
                break;
            }
            case INT16: {
                auto const p{static_cast<int16_t *>(this->get_raw_buffer())};
                PRINT_BARRAY(p, len(), 0);
                break;
            }
            case INT32: {
                auto const p{static_cast<int32_t *>(this->get_raw_buffer())};
                PRINT_BARRAY(p, len(), 0);
                break;
            }
            case INT64: {
                auto const p{static_cast<int64_t *>(this->get_raw_buffer())};
                PRINT_BARRAY(p, len(), 0);
                break;
            }
            case FLOAT32: {
                auto const p{static_cast<float *>(this->get_raw_buffer())};
                PRINT_BARRAY(p, len(), 3);
                break;
            }
            case FLOAT64: {
                auto const p{static_cast<double *>(this->get_raw_buffer())};
                PRINT_BARRAY(p, len(), 5);
                break;
            }
            case INVALID:
                return;
        }

    }
}
