//
// Created by sriram on 1/25/25.
//

#include "brarray.h"
#include <iostream>
#include <iomanip>
#include "math_dis.h"
#include "allocator.h"
#include "computation_graph.h"

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
        void *buffer{nullptr};
        Allocator *allocator{nullptr};
        size_t columns{0};
        size_t rows{0};
        size_t dtype_size{0};

        BufferContainer() = default;

        BufferContainer(size_t const rows, size_t const columns, Allocator *p_alloc,
                        size_t const dtype_size): allocator(p_alloc), rows(rows), dtype_size(dtype_size) {
            this->columns = allocator->calloc(&buffer, rows, columns, dtype_size) / dtype_size;
        }

        ~BufferContainer() {
            allocator->free(buffer);
        }

        BufferContainer(const BufferContainer &other): allocator(other.allocator), columns(other.columns),
                                                       rows(other.rows), dtype_size(other.dtype_size) {
            allocator->calloc(&buffer, rows, columns, dtype_size);
            allocator->mem_copy(buffer, other.buffer, rows * columns * dtype_size);
        }

        BufferContainer &operator=(const BufferContainer &other) {
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
        std::shared_ptr<ActivationNode> node = nullptr;

        ArrayImpl(Device const device, Dtype const dtype, std::vector<size_t> const &shape): shape(shape),
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

        [[nodiscard]] void *buffer() const {
            return buffer_container->buffer;
        }

        ArrayImpl() = default;

        ArrayImpl(const ArrayImpl &) = delete;

        ArrayImpl &operator=(const ArrayImpl &) = delete;

        void multiply(const ArrayImpl *other,
                      ArrayImpl *dest,
                      const std::vector<size_t> &common_shape,
                      const std::vector<size_t> &this_stride,
                      const std::vector<size_t> &other_stride) const {

            // std::cout << this->offset << std::endl;

            this->m_dispatcher->hadamard_product(
                static_cast<char *>(this->buffer()) + this->offset,
                static_cast<char *>(other->buffer()) + other->offset,
                static_cast<char *>(dest->buffer()) + dest->offset,
                common_shape.data(),
                common_shape.size(),
                this_stride.data(),
                other_stride.data(),
                dest->buffer_container->columns,
                dtype);
        }

        void add(const ArrayImpl *other,
                 ArrayImpl *dest,
                 const std::vector<size_t> &common_shape,
                 const std::vector<size_t> &this_stride,
                 const std::vector<size_t> &other_stride) const {

            this->m_dispatcher->element_wise_add(
                static_cast<char *>(this->buffer()) + this->offset,
                static_cast<char *>(other->buffer()) + other->offset,
                static_cast<char *>(dest->buffer()) + dest->offset,
                common_shape.data(),
                common_shape.size(),
                this_stride.data(),
                other_stride.data(),
                dest->buffer_container->columns,
                dtype);
        }
    };

    Brarray::Brarray(Device const device, Dtype const dtype, std::vector<size_t> const &shape): impl(
        std::make_unique<ArrayImpl>(device, dtype, shape)) {
        is_invalid(dtype);
    }

    void Brarray::requires_grad(const bool state) {
        if (state && !requires_grad()) {
            impl->node = std::make_shared<ActivationNode>(
                *this,
                std::initializer_list<std::shared_ptr<AutogradNode>>{}, true);
        }

        if (!state && requires_grad()) {
            impl->node = nullptr;
        }
    }

    bool Brarray::requires_grad() const {
        return impl->node != nullptr;
    }

    void Brarray::retain_grad(const bool state) {
        if (!requires_grad()) throw std::runtime_error("brarray must require gradients in order to retain them");
       if (!impl->node->retain_grad && state) {
           impl->node->retain_grad = true;
       } else if(impl->node->retain_grad && !state) {
           impl->node->retain_grad = false;
       }
    }

    bool Brarray::retain_grad() const{
        if (!requires_grad()) throw std::runtime_error("brarray must require gradients in order to retain them");
        return impl->node->retain_grad;
    }

    Brarray &Brarray::get_gradient() {
        if (!requires_grad()) throw std::runtime_error("brarray must require gradients in order to contain them");
        if (!retain_grad()) throw std::runtime_error("brarray does not retain gradients");
        return impl->node->gradient;
    }

    void Brarray::backwards() {
        if (!is_scalar_equivalent()) throw std::runtime_error("currently backwards can only be calculate on scalars");
        back_propagate(this->impl->node);
    }

    Brarray::Brarray(const Device device, const Dtype dtype, std::vector<size_t> const &shape, const void *ptr): impl(
        std::make_unique<ArrayImpl>(device, dtype, shape)) {
        const size_t &total_cols{impl->buffer_container->columns}; // includes padding
        const size_t &total_rows{impl->buffer_container->rows};
        const size_t &requested_cols{shape[shape.size() - 1]};
        const size_t &dtype_size{impl->buffer_container->dtype_size};

        auto *buff{static_cast<char *>(get_raw_buffer())};
        auto *char_ptr{static_cast<const char *>(ptr)};

        for (size_t i{0}; i < total_rows; ++i) {
            impl->buffer_container->allocator->mem_copy(
                &buff[i * total_cols * dtype_size],
                &char_ptr[i * requested_cols * dtype_size],
                dtype_size * requested_cols);
        }
    }

    Brarray::Brarray(Brarray &&other) noexcept: impl(std::move(other.impl)) {
        other.impl = nullptr;
    }

    Brarray &Brarray::operator=(Brarray &&other) noexcept {
        if (this == &other) return *this;
        this->impl = std::move(other.impl);
        other.impl = nullptr;
        return *this;
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

    Brarray::Brarray(): impl(std::make_unique<ArrayImpl>()) {
    }

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

    Brarray Brarray::shared_copy() const {
        Brarray ret;
        ret.impl->offset = impl->offset;
        ret.impl->shape = impl->shape;
        ret.impl->stride = impl->stride;
        ret.impl->device = impl->device;
        ret.impl->dtype = impl->dtype;
        ret.impl->m_dispatcher = impl->m_dispatcher;
        ret.impl->buffer_container = impl->buffer_container;
        return ret;
    }


    void check_dtype(const Brarray &b1, const Brarray &b2) {
        is_invalid(b1.get_dtype());
        if (b1.get_dtype() != b2.get_dtype()) throw std::runtime_error("dtypes do not match");
    }

    bool check_shapes(const Brarray &b1, const Brarray &b2) {
        if (b1.get_shape().empty()) throw std::runtime_error("brarray is empty");
        if (b1.get_shape() != b2.get_shape()) return false;
        return true;
    }

    void check_devices(const Brarray &b1, const Brarray &b2) {
        if (b1.get_device() != b2.get_device()) throw std::runtime_error("devices do not match");
    }

    void leading_expand_dims(const int index, std::vector<size_t> &shape) {
        if (shape.empty()) throw std::runtime_error("shape must not be empty");
        if (static_cast<size_t>(index) > shape.size() - 1) throw std::runtime_error("index is not a leading dimension");
        std::vector<size_t> new_shape(shape.size() + 1, 1);
        shape.insert(shape.begin() + index, 1);
    }

    bool ican_broadcast(const std::vector<size_t> &base_shape, const Brarray &other) {
        const std::vector<size_t> &other_shape{other.get_shape()};
        for (size_t i{1}; i < other_shape.size() + 1; ++i) {
            if (base_shape[base_shape.size() - i] == other_shape[other_shape.size() - i]) continue;
            if (other_shape[other_shape.size() - i] == 1) continue;
            return false;
        }

        return true;
    }

    bool can_broadcast(const Brarray &arr_one, const Brarray &arr_two) {
        const std::vector<size_t> &one_shape{arr_one.get_shape()};
        const std::vector<size_t> &two_shape{arr_two.get_shape()};

        const size_t small{std::min(arr_one.get_shape().size(), arr_two.get_shape().size())};

        for (size_t i{1}; i < small + 1; ++i) {
            if (one_shape[one_shape.size() - i] == two_shape[two_shape.size() - i]) continue;
            if (one_shape[one_shape.size() - i] == 1 || two_shape[two_shape.size() - i] == 1) continue;
            return false;
        }

        return true;
    }

    std::vector<size_t> calculate_broadcasted_shape(const Brarray &arr_one, const Brarray &arr_two) {
        const std::vector<size_t> &one_shape{arr_one.get_shape()};
        const std::vector<size_t> &two_shape{arr_two.get_shape()};

        const size_t small{std::min(arr_one.get_shape().size(), arr_two.get_shape().size())};
        const std::vector<size_t> &short_shape{small == one_shape.size() ? one_shape : two_shape};
        const std::vector<size_t> &long_shape{small == one_shape.size() ? two_shape : one_shape};

        std::vector<size_t> new_shape(long_shape.size(), 0);

        size_t i{1};
        for (; i <= short_shape.size(); ++i)
            new_shape[new_shape.size() - i] = std::max(
                short_shape[short_shape.size() - i],
                long_shape[long_shape.size() - i]);
        for (; i <= long_shape.size(); ++i) new_shape[new_shape.size() - i] = long_shape[long_shape.size() - i];

        return new_shape;
    }

    std::vector<size_t> calculate_broadcasted_stride(
        const std::vector<size_t> &broadcast_shape,
        std::vector<size_t> original_shape,
        const size_t columns) {
        while (original_shape.size() < broadcast_shape.size()) {
            leading_expand_dims(0, original_shape);
        }

        std::vector<size_t> new_stride(original_shape.size(), 1);
        init_stride(original_shape, new_stride, columns);

        for (size_t i = 0; i < original_shape.size(); ++i) {
            if (original_shape[i] == 1) new_stride[i] = 0;
        }

        return new_stride;
    }

    struct ElementWiseData {
        std::vector<size_t> shape{};
        std::vector<size_t> stride_one{};
        std::vector<size_t> stride_two{};
    };

    ElementWiseData validate_element_wise(
        const Brarray &arr_one,
        const Brarray &arr_two,
        const size_t last_stride_one,
        const size_t last_stride_two) {

        check_dtype(arr_one, arr_two);
        check_devices(arr_one, arr_two);

        ElementWiseData ret;

        if (!check_shapes(arr_one, arr_two)) {
            if (can_broadcast(arr_one, arr_two)) {
                ret.shape = calculate_broadcasted_shape(arr_one, arr_two);
                ret.stride_one = calculate_broadcasted_stride(
                    ret.shape,
                    arr_one.get_shape(),
                    last_stride_one);

                ret.stride_two = calculate_broadcasted_stride(
                    ret.shape,
                    arr_two.get_shape(),
                    last_stride_two);
                return ret;
            }

            throw std::runtime_error("cannot perform element wise operations on invalid shapes");
            // TODO make more descriptive
        }

        ret.shape = arr_one.get_shape();
        ret.stride_one = arr_one.get_stride();
        ret.stride_two = arr_two.get_stride();

        return ret;
    }

    ElementWiseData ivalidate_element_wise(
        const Brarray &base_array,
        const Brarray &other,
        const size_t last_stride_two) {

        check_dtype(base_array, other);
        check_devices(base_array, other);

        ElementWiseData ret;

        if (!check_shapes(base_array, other)) {
            if (ican_broadcast(base_array.get_shape(), other)) {
                ret.shape = base_array.get_shape();
                ret.stride_one = base_array.get_stride();

                ret.stride_two = calculate_broadcasted_stride(
                    ret.shape,
                    other.get_shape(),
                    last_stride_two);
                return ret;
            }

            throw std::runtime_error("cannot perform element wise operations on invalid shapes");
        }

        ret.shape = base_array.get_shape();
        ret.stride_one = base_array.get_stride();
        ret.stride_two = other.get_stride();
        return ret;
    }

    Brarray Brarray::operator*(const Brarray &other) const {
        return mult(*this, other, true);
    }

    Brarray Brarray::operator+(const Brarray &other) const {
        return add(*this, other, true);
    }

    Brarray mult(const Brarray &input, const Brarray &other, const bool track_gradients) {
        if (track_gradients) {}

        const auto [shape, stride_one, stride_two]{
            validate_element_wise(
                input,
                other,
                input.impl->buffer_container->columns,
                other.impl->buffer_container->columns)};

        Brarray result(input.get_device(), input.get_dtype(), shape);

        input.impl->multiply(
            other.impl.get(),
            result.impl.get(),
            shape,
            stride_one,
            stride_two);

        return result;
    }

    Brarray add(const Brarray &input, const Brarray &other, bool track_gradients) {
        if (track_gradients) {}

        const auto [shape, stride_one, stride_two]{
            validate_element_wise(
                input,
                other,
                input.impl->buffer_container->columns,
                other.impl->buffer_container->columns)};

        Brarray result(input.get_device(), input.get_dtype(), shape);

        input.impl->add(
            other.impl.get(),
            result.impl.get(),
            shape,
            stride_one,
            stride_two);

        return result;
    }


    void imult(Brarray &input, const Brarray &other) {
        // TODO ADD warning for requires grads
        const auto [shape, stride_one, stride_two]{
            ivalidate_element_wise(
                input,
                other,
                other.impl->buffer_container->columns)};

        input.impl->multiply(
            other.impl.get(),
            input.impl.get(),
            shape,
            stride_one,
            stride_two);
    }

    void iadd(Brarray &input, const Brarray &other) {
        // TODO ADD warning for requires grads
        const auto [shape, stride_one, stride_two]{
            ivalidate_element_wise(
                input,
                other,
                other.impl->buffer_container->columns)};

        input.impl->add(
            other.impl.get(),
            input.impl.get(),
            shape,
            stride_one,
            stride_two);
    }

    template<typename Dtype>
    Brarray scalar_to_brarray(const Brarray &arr, Dtype value) {
        const core::Dtype current_dtype{arr.get_dtype()};
        if (current_dtype != get_dtype_from_type<Dtype>::type && current_dtype != FLOAT32 && current_dtype != FLOAT64) {
            throw std::runtime_error("dtype of scalar variable does not match brarray dtype"); //TODO make more descriptive
        }

        if (current_dtype == get_dtype_from_type<Dtype>::type) {
            // std::cout << value << std::endl;
            return Brarray(arr.get_device(), arr.get_dtype(), {1}, std::vector{value});
        }

        if (current_dtype == FLOAT32) {
            float casted_value{static_cast<float>(value)};
            return Brarray(arr.get_device(), arr.get_dtype(), {1}, std::vector{casted_value});
        }

        double casted_value{static_cast<double>(value)};
        return Brarray(arr.get_device(), arr.get_dtype(), {1}, std::vector{casted_value});
    }

    template <typename Dtype>
    void imult(Brarray &input, const Dtype other) {
        imult(input, scalar_to_brarray(input, other));
    }

    template <typename Dtype>
    void iadd(Brarray &input, const Dtype other) {
        iadd(input, scalar_to_brarray(input, other));
    }

    template<typename Dtype>
    Brarray Brarray::operator*(Dtype other) const {
        const Brarray& th{*this};
        return mult(th, scalar_to_brarray(th, other), true);
    }

    template<typename Dtype>
    Brarray Brarray::operator+(Dtype other) const {
        const Brarray& th{*this};
        return add(th, scalar_to_brarray(th, other), true);
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
        const void *data) {
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
        if (matrix.get_shape()[1] != vector.get_shape()[0])
            throw std::runtime_error(
                "vector and matrix have different columns lengths");
        if (matrix.get_shape()[0] != this->get_shape()[0])
            throw std::runtime_error(
                "the current brarray must be of shape (matrix.get_shape()[0])");


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

        if (current_shape.size() == 1) {
            // sadly deep copy is required to maintain alignment
            // std::cout << "deep copy" << std::endl;
            Brarray scalar(impl->device, impl->dtype, {1});
            scalar.impl->buffer_container->allocator->mem_copy(
                scalar.get_raw_buffer(),
                static_cast<char *>(this->get_raw_buffer()) + index * dtype_to_bytes(this->impl->dtype),
                dtype_to_bytes(this->impl->dtype)
            );

            return scalar;
        }

        Brarray ret;

        ret.impl->stride = impl->stride;
        ret.impl->shape = impl->shape;
        ret.impl->stride.erase(ret.impl->stride.begin());
        ret.impl->shape.erase(ret.impl->shape.begin());
        ret.impl->dtype = impl->dtype;
        ret.impl->buffer_container = impl->buffer_container;
        ret.impl->m_dispatcher = impl->m_dispatcher;
        ret.impl->offset = impl->offset;
        ret.impl->offset += impl->stride[0] * index * dtype_to_bytes(this->impl->dtype);
        ret.impl->device = impl->device;

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

    void Brarray::to_string_helper(std::stringstream &ss, const Brarray &br, const std::string &gap) const {
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
        is_invalid(b.get_dtype());
        const std::string str = b.to_string();
        outs << b.generate_description();
        outs << str;
        return outs;
    }

    INSTANTIATE_OPERATOR(Brarray::operator*);
    INSTANTIATE_OPERATOR(Brarray::operator+);
    INSTANTIATE_INPLACE_OPERATOR(imult);
    INSTANTIATE_INPLACE_OPERATOR(iadd);
}
