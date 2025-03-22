//
// Created by sriram on 1/25/25.
//

#include "brarray.h"
#include <iostream>
#include <iomanip>
#include <stack>

#include "math_dis.h"
#include "allocator.h"

namespace cobraml::core {
    void init_stride(std::vector<size_t> const &shape,
                    std::vector<size_t> &stride,
                    size_t const column_count) {

        stride[stride.size() - 1] = 1;

        if (shape.size() == 1)
            return;

        size_t sum = column_count;

        for (long i = static_cast<long>(shape.size() - 2); i >= 0; --i) {
            stride[i] = sum;
            sum *= shape[i];
        }
    }

    struct BufferContainer {
        void * buffer{nullptr};
        Allocator * allocator{nullptr};
        size_t columns{0};
        size_t rows{0};
        size_t dtype_size{0};

        BufferContainer() = default;
        BufferContainer(size_t const rows, size_t const columns, Allocator * p_alloc, size_t const dtype_size): allocator(p_alloc), rows(rows), dtype_size(dtype_size){
            this->columns = allocator->calloc(&buffer, rows, columns, dtype_size);
        }

        ~BufferContainer() {
            allocator->free(buffer);
        }

        BufferContainer(const BufferContainer & other):
        allocator(other.allocator), columns(other.columns), rows(other.rows), dtype_size(other.dtype_size){
            allocator->calloc(&buffer, rows, columns, dtype_size);
            allocator->mem_copy(buffer, other.buffer, rows * columns * dtype_size);
        }

        BufferContainer &operator=(const BufferContainer & other) {

            if (this == &other) return *this;

            if (other.rows != rows && other.columns != columns) {
                allocator->free(buffer);
                other.allocator->calloc(&buffer, other.rows, other.columns, other.dtype_size);
            }

            allocator = other.allocator;
            columns = other.columns;
            rows = other.rows;
            dtype_size = other.dtype_size;

            allocator->mem_copy(buffer, other.buffer, rows * columns * dtype_size);

            return *this;
        }
    };

    struct Brarray::ArrayImpl {
        size_t offset = 0;
        std::vector<size_t> shape{};
        std::vector<size_t> stride{};
        Device device = CPU;
        Dtype dtype = INVALID;
        Math *m_dispatcher = nullptr;
        std::shared_ptr<BufferContainer> buffer_container = nullptr;

        ArrayImpl(Device const device, Dtype const dtype, std::vector<size_t> const &shape):
            shape(shape),
            stride(shape.size(), 0),
            device(device),
            dtype(dtype),
            m_dispatcher(get_math_kernels(device)) {

            is_invalid(dtype);
            if (shape.empty()) throw std::runtime_error("cannot initialize Brarray with an empty shape");

            size_t rows{1};
            size_t columns = shape[shape.size() - 1];

            if (shape.size() > 1) {
                for (size_t i{0}; i < shape.size() - 1; ++i) {
                    rows *= shape[i];
                }
            }

            buffer_container = std::make_shared<BufferContainer>(
                rows, columns, get_allocator(device), dtype_to_bytes(dtype));

            columns = buffer_container->columns;
            init_stride(shape, stride, columns);
        }

        [[nodiscard]] void * buffer() const {
            return buffer_container->buffer;
        }

        ArrayImpl() = default;
        ArrayImpl(const ArrayImpl &) = delete;
        ArrayImpl &operator=(const ArrayImpl &) = delete;
    };

    Brarray::Brarray(Device const device, Dtype const dtype, std::vector<size_t> const &shape): impl(
        std::make_unique<ArrayImpl>(device, dtype, shape)) {
        is_invalid(dtype);
    }

    Dtype Brarray::get_dtype() const {
        return impl->dtype;
    }

    Device Brarray::get_device() const {
        return impl->device;
    }

    std::vector<size_t> Brarray::get_shape() const {
        return impl->shape;
    }

    std::vector<size_t> Brarray::get_stride() const {
        return impl->stride;
    }

    void *Brarray::get_raw_buffer() const {
        if (impl->buffer_container == nullptr) {
            throw std::runtime_error("data buffer is null");
        }

        return static_cast<char *>(impl->buffer()) + impl->offset;
    }

    Brarray::~Brarray() = default;

    Brarray::Brarray(): impl(std::make_unique<ArrayImpl>()) {}

    Brarray::Brarray(const Brarray &other) : impl(std::make_unique<ArrayImpl>()) {
        impl->offset = other.impl->offset;
        impl->shape = other.impl->shape;
        impl->stride = other.impl->stride;
        impl->device = other.impl->device;
        impl->dtype = other.impl->dtype;
        impl->m_dispatcher = other.impl->m_dispatcher;
        impl->buffer_container = std::make_shared<BufferContainer>(*other.impl->buffer_container);
    }

    Brarray &Brarray::operator=(const Brarray &other) {
        if (this == &other)
            return *this;

        impl->offset = other.impl->offset;
        impl->shape = other.impl->shape;
        impl->stride = other.impl->stride;
        impl->device = other.impl->device;
        impl->dtype = other.impl->dtype;
        impl->m_dispatcher = other.impl->m_dispatcher;
        impl->buffer_container = std::make_shared<BufferContainer>(*other.impl->buffer_container);

        return *this;
    }

    bool Brarray::is_matrix() const {
        return get_shape().size() == 2;
    }

    bool Brarray::is_vector() const {
        return get_shape().size() == 1;
    }

    bool Brarray::is_scalar_equivalent() const {
        if (is_vector()) return get_shape()[0] == 1;
        return false;
    }

    void Brarray::gemv(
        const Brarray &matrix,
        const Brarray &vector,
        size_t const rows,
        size_t const columns,
        const void *alpha,
        const void *beta) {

        if (!vector.is_vector()) throw std::runtime_error("supplied 'vector' is not a vector");
        if (this->is_vector()) throw std::runtime_error("the current brarray is not a vector");
        if (matrix.is_matrix()) throw std::runtime_error("supplied 'matrix' is not a matrix");
        if (matrix.get_shape()[1] != vector.get_shape()[0]) throw std::runtime_error("vector and matrix have different columns lengths");
        if (matrix.get_shape()[0] != this->get_shape()[0]) throw std::runtime_error("the current brarray must be of shape (matrix.get_shape()[0])");


        if (matrix.get_device() != vector.get_device() || this->get_device() != matrix.get_device()) {
            throw std::runtime_error("vector, matrix and the current brarray are not on the same device");
        }

        if (matrix.get_dtype() != vector.get_dtype() || matrix.get_dtype() != this->get_dtype()) {
            throw std::runtime_error("vector, matrix and current brarray have different dtypes");
        }

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

    void Brarray::reassign_vector(const void *source) {
        if (!is_vector()) {
            throw std::runtime_error("can only reassign values for valid vectors");
        }

        this->impl->buffer_container->allocator->mem_copy(
            this->get_raw_buffer(),
            source,
            this->get_shape()[0] * dtype_to_bytes(this->get_dtype()));
    }

    Brarray Brarray::operator[](size_t const index) const {

        std::vector<size_t> const &current_shape = this->get_shape();
        if (index >= current_shape[0]) throw std::out_of_range("index out of bounds");

        Brarray ret;
        ret.impl = std::move(
            std::make_unique<ArrayImpl>(this->get_device(), this->get_dtype(), std::vector<size_t>{1}));

        ret.impl->stride = impl->stride;
        ret.impl->buffer_container = impl->buffer_container;
        ret.impl->m_dispatcher = impl->m_dispatcher;
        ret.impl->shape = impl->shape;
        ret.impl->offset = impl->offset;

        ret.impl->offset += ret.impl->stride[0] * index * dtype_to_bytes(this->impl->dtype);

        if (current_shape.size() == 1) { // only really occurs when indexing a scalar with 0
            ret.impl->stride = {1};
            ret.impl->shape = {1};
            return ret;
        }

        ret.impl->stride.erase(ret.impl->stride.begin());
        ret.impl->shape.erase(ret.impl->shape.begin());

        return ret;
    }

    std::string shape_to_str(std::vector<size_t> const &shp, std::stringstream &ss) {
        ss << "[";

        for (size_t i{0}; i < shp.size() - 1; ++i) {
            ss << std::to_string(i) << ", ";
        }

        if (!shp.empty()) ss << std::to_string(shp[shp.size() - 1]);
        ss << "]";

        return ss.str();
    }

    std::string Brarray::generate_description() const {
        std::stringstream ss;
        ss << "############## Details ##############\n";
        ss << "Shape: " << shape_to_str(this->get_shape(), ss) << "\n";
        ss << "Device: " << device_to_string(this->get_device()) << "\n";
        ss << "Dtype: " << dtype_to_string(this->get_dtype()) << "\n";
        ss << "#####################################\n";

        return ss.str();
    }

    std::string Brarray::to_string() const {
        if (this->get_dtype() == INVALID) {
            throw std::runtime_error("cannot convert Brarray with invalid dtype to a string");
        }

        std::stringstream ss;

        size_t vector_count{1};
        for (size_t i{0}; i < get_shape().size() - 1; ++i) {
            vector_count *= get_shape()[i];
        }

        std::vector<Brarray> preallocated_vec;
        std::vector<std::string> preallocated_str_vec;

        preallocated_vec.reserve(vector_count);
        preallocated_str_vec.reserve(vector_count);

        std::queue temps(preallocated_vec);
        temps.push(*this);

        std::stack brackets(preallocated_str_vec);

        std::string gap;

        while (!temps.empty()) {
            ss << gap;
            Brarray& current = temps.front();

            if (current.is_vector()) {
                switch (this->impl->dtype) {
                    case INT8: {
                        print_vector<int8_t>(current.get_raw_buffer(), current.get_shape()[1], ss);
                        break;
                    }
                    case INT16: {
                        print_vector<int16_t>(current.get_raw_buffer(), current.get_shape()[1], ss);
                        break;
                    }
                    case INT32: {
                        print_vector<int32_t>(current.get_raw_buffer(), current.get_shape()[1], ss);
                        break;
                    }
                    case INT64: {
                        print_vector<int64_t>(current.get_raw_buffer(), current.get_shape()[1], ss);
                        break;
                    }
                    case FLOAT32: {
                        print_vector<float>(current.get_raw_buffer(), current.get_shape()[1], ss);
                        break;
                    }
                    case FLOAT64: {
                        print_vector<double>(current.get_raw_buffer(), current.get_shape()[1], ss);
                        break;
                    }
                    case INVALID:
                        throw std::runtime_error("cannot convert Brarray with invalid dtype to a string");;
                }
                ss << "\n";
                temps.pop();
                continue;
            }

            ss << "[\n";
            brackets.emplace(gap + "]\n");
            gap += " ";

            for (size_t i{0}; i < current.get_shape()[0]; ++i) temps.push(current[i]);
            temps.pop();
        }

        while (!brackets.empty()) {
            ss << brackets.top();
            brackets.pop();
        }

        return ss.str();
    }

    std::ostream &operator<<(std::ostream &outs, const Brarray &b) {
        const std::string str = b.to_string();
        outs << b.generate_description();
        outs << str;
        return outs;
    }
}
