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

        size_t sum{column_count};

        for (long i{static_cast<long>(shape.size()) - 2}; i >= 0; --i) {
            stride[static_cast<size_t>(i)] = sum;
            sum *= shape[static_cast<size_t>(i)];
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
            this->columns = allocator->calloc(&buffer, rows, columns, dtype_size) / dtype_size;
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
            if (other.rows * other.columns * other.dtype_size != rows * columns * dtype_size) {
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
            if (shape[0] == 0) throw std::runtime_error("dimensions in shape cannot be zero");
            size_t columns = shape[shape.size() - 1];

            if (shape.size() > 1) {
                for (size_t i{0}; i < shape.size() - 1; ++i) {
                    if (shape[i] == 0) throw std::runtime_error("dimensions in shape cannot be zero");

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

    Brarray::Brarray(const Device device, const Dtype dtype, std::vector<size_t> const &shape, const void *ptr):
        impl(std::make_unique<ArrayImpl>(device, dtype, shape)) {

        const size_t & total_cols{impl->buffer_container->columns}; // includes padding
        const size_t & total_rows{impl->buffer_container->rows};
        const size_t & requested_cols{shape[shape.size() - 1]};
        const size_t & dtype_size{impl->buffer_container->dtype_size};

        auto * buff{static_cast<char *>(get_raw_buffer())};
        auto * char_ptr{static_cast<const char *>(ptr)};

        for (size_t i{0}; i < total_rows; ++i) {
            impl->buffer_container->allocator->mem_copy(
                &buff[i * total_cols * dtype_size],
                &char_ptr[i * requested_cols * dtype_size],
                dtype_size * requested_cols);
        }
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

    const void *Brarray::validated_get_data(
        Dtype const dtype_vec,
        Dtype const provided,
        size_t const shape,
        std::vector<size_t> const &provided_shape,
        const void * data) {

        is_invalid(provided);
        if (dtype_vec != provided)
            throw std::runtime_error(
                "Template dtype T does not match brarray dtypes");

        if (provided_shape.empty()) throw std::runtime_error("an empty shape was provided");

        size_t prod{1};
        for (const auto &dim: provided_shape) prod *= dim;

        if (shape == 0) throw std::runtime_error("provided data is empty");

        if (prod != shape) throw std::runtime_error("given shape cannot compensate provided data");

        return data;
    }


    void Brarray::gemv(
        const Brarray &matrix,
        const Brarray &vector,
        const void *alpha,
        const void *beta) {

        if (!vector.is_vector()) throw std::runtime_error("supplied 'vector' is not a vector");
        if (!this->is_vector()) throw std::runtime_error("the current brarray is not a vector");
        if (!matrix.is_matrix()) throw std::runtime_error("supplied 'matrix' is not a matrix");
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
            matrix.get_shape()[0],
            matrix.get_shape()[1],
            matrix.get_stride()[0],
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

        is_invalid(this->get_dtype());
        std::vector<size_t> const &current_shape = this->get_shape();
        if (current_shape.empty()) throw std::out_of_range("index out of bounds");
        if (index >= current_shape[0]) throw std::out_of_range("index out of bounds");

        Brarray ret;
        size_t jump;

        if (current_shape.size() == 1) {
            ret.impl->stride = {1};
            ret.impl->shape = {1};
            jump = 1;
        }else {
            ret.impl->stride = impl->stride;
            ret.impl->shape = impl->shape;
            jump = ret.impl->stride[0];
            ret.impl->stride.erase(ret.impl->stride.begin());
            ret.impl->shape.erase(ret.impl->shape.begin());
        }

        ret.impl->dtype = impl->dtype;
        ret.impl->buffer_container = impl->buffer_container;
        ret.impl->m_dispatcher = impl->m_dispatcher;
        ret.impl->offset = impl->offset;
        ret.impl->offset += jump * index * dtype_to_bytes(this->impl->dtype);

        return ret;
    }

    void shape_to_str(std::vector<size_t> const &shp, std::stringstream &ss) {
        ss << "[";

        for (size_t i{0}; i < shp.size() - 1; ++i) {
            ss << std::to_string(shp[i]) << ", ";
        }

        if (!shp.empty()) ss << std::to_string(shp[shp.size() - 1]);
        ss << "]";
    }

    std::string Brarray::generate_description() const {
        std::stringstream ss;
        ss << "############## Details ##############\n";
        ss << "Shape: ";
        shape_to_str(this->get_shape(), ss);
        ss << "\n";
        ss << "Device: " << device_to_string(this->get_device()) << "\n";
        ss << "Dtype: " << dtype_to_string(this->get_dtype()) << "\n";
        ss << "#####################################\n";

        return ss.str();
    }

    void Brarray::to_string_helper(std::stringstream & ss, const Brarray & br, const std::string & gap) const{
        if (br.is_vector()) {
            ss << gap;
            switch (br.get_dtype()) {
                case INT8: {
                    print_vector<int8_t>(br.get_raw_buffer(), br.get_shape()[0], ss);
                    break;
                }
                case INT16: {
                    print_vector<int16_t>(br.get_raw_buffer(), br.get_shape()[0], ss);
                    break;
                }
                case INT32: {
                    print_vector<int32_t>(br.get_raw_buffer(), br.get_shape()[0], ss);
                    break;
                }
                case INT64: {
                    print_vector<int64_t>(br.get_raw_buffer(), br.get_shape()[0], ss);
                    break;
                }
                case FLOAT32: {
                    print_vector<float>(br.get_raw_buffer(), br.get_shape()[0], ss);
                    break;
                }
                case FLOAT64: {
                    print_vector<double>(br.get_raw_buffer(), br.get_shape()[0], ss);
                    break;
                }
                case INVALID:
                    throw std::runtime_error("cannot convert Brarray with invalid dtype to a string");;
            }
            ss << "\n";
            return;
        }

        ss << gap;
        ss << "[\n";
        for (size_t i{0}; i < br.get_shape()[0]; ++i) to_string_helper(ss, br[i], gap + "   ");
        ss << gap;
        ss << "]\n";
    }

    std::string Brarray::to_string() const {
        std::stringstream ss;
        to_string_helper(ss, *this, "");
        return ss.str();
    }

    std::ostream &operator<<(std::ostream &outs, const Brarray &b) {
        const std::string str = b.to_string();
        outs << b.generate_description();
        outs << str;
        return outs;
    }
}
