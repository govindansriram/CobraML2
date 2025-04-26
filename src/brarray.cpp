//
// Created by sriram on 1/25/25.
//

#include "brarray.h"
#include <iostream>
#include <iomanip>
#include "math_dis.h"
#include "allocator.h"
#include "computation_graph.h"
#include "layers/element_wise.h"
#include "instantiate.h"

namespace cobraml::core {
    /**
     * calculate the stride of the brarray
     * @param shape the shape of the brarray
     * @param stride the stride that will be overwritten must be prefilled with the same amount of dims as shape
     * @param column_count the stride of the 2nd dimension in the brarray which is the amount of columns. This value
     * may not be the same as the shape value, if padding is added by the allocator
     */
    void init_stride(std::vector<size_t> const &shape,
                     std::vector<size_t> &stride,
                     size_t const column_count) {

        if (shape.size() != stride.size()) throw std::runtime_error("shape and stride need to be the same length");

        stride[stride.size() - 1] = 1;

        if (shape.size() == 1)
            return;

        size_t sum{column_count};

        for (long i{static_cast<long>(shape.size()) - 2}; i >= 0; --i) {
            stride[static_cast<size_t>(i)] = sum;
            sum *= shape[static_cast<size_t>(i)];
        }
    }

    void shape_to_str(std::vector<size_t> const &shp, std::stringstream &ss) {
        ss << "[";

        for (size_t i{0}; i < shp.size() - 1; ++i) {
            ss << std::to_string(shp[i]) << ", ";
        }

        if (!shp.empty()) ss << std::to_string(shp[shp.size() - 1]);
        ss << "]";
    }

    struct BufferContainer {
        void *buffer{nullptr};
        Allocator *allocator{nullptr};
        Device device{CPU};
        Dtype dtype{INVALID};
        size_t true_column_len{0};

        BufferContainer(const std::vector<size_t> &shape, Device const device, Dtype const dtype):
            allocator(get_allocator(device)), device(device), dtype(dtype) {
            const size_t allocated_bytes{allocator->calloc(&buffer, shape, dtype)};
            const size_t rows{calculate_total_rows(shape)};
            true_column_len = allocated_bytes / rows / dtype_to_bytes(dtype);
        }

        ~BufferContainer() {
            allocator->free(buffer);
        }

        [[nodiscard]] size_t get_true_column_len() const {
            return true_column_len;
        }

        /**
         * use this constructor for deep copies where the stride for the new buffer is the same
         * @param other
         * @param offset_bytes
         * @param new_shape
         */
        BufferContainer(
            const BufferContainer * other,
            const size_t offset_bytes,
            const std::vector<size_t> &new_shape):
                allocator(other->allocator),
                device(other->device),
                dtype(other->dtype),
                true_column_len(other->true_column_len){

            if (other->buffer == nullptr) throw std::runtime_error("cannot deepcopy invalid buffer");
            const auto data_ptr{static_cast<char *>(other->buffer) + offset_bytes};
            const size_t allocated_bytes{allocator->malloc(&buffer, new_shape, dtype)};

            allocator->mem_copy(
                    buffer,
                    data_ptr,
                    allocated_bytes,
                    DEVICE_TO_DEVICE);
        }

        BufferContainer(const BufferContainer &other) = delete;
        BufferContainer &operator=(const BufferContainer &other) = delete;
        BufferContainer() = delete;
    };

    struct Brarray::ArrayImpl {
        size_t offset{0};
        std::vector<size_t> shape{};
        std::vector<size_t> stride{};
        Device device = CPU;
        Dtype dtype = INVALID;
        Math *m_dispatcher = nullptr;
        std::shared_ptr<BufferContainer> buffer_container = nullptr;
        std::shared_ptr<ActivationNode> node = nullptr;
        size_t total_rows{0};

        ArrayImpl(Device const device, Dtype const dtype, std::vector<size_t> const &shape): shape(shape),
            stride(shape.size(), 0),
            device(device),
            dtype(dtype),
            m_dispatcher(get_math_kernels(device)),
            total_rows(calculate_total_rows(shape)){

            is_invalid(dtype);

            buffer_container = std::make_shared<BufferContainer>(shape, device, dtype);
            init_stride(shape, stride, buffer_container->get_true_column_len());
        }

        ArrayImpl(const Device device, const Dtype dtype, const MemoryDirection direction,
            const std::vector<size_t> &shape, const size_t source_stride, const void *ptr)
                : ArrayImpl(device, dtype, shape){

            const size_t column_count{shape[shape.size() - 1]};
            const size_t elements{calculate_total_rows(shape) * column_count};

            buffer_container->allocator->strided_mem_copy(
                get_raw_buffer(),
                ptr,
                elements * dtype_to_bytes(dtype),
                direction,
                column_count * dtype_to_bytes(dtype),
                (buffer_container->get_true_column_len() - column_count) * dtype_to_bytes(dtype),
                (source_stride - column_count) * dtype_to_bytes(dtype));
        }

        [[nodiscard]] void *buffer() const {
            return buffer_container->buffer;
        }

        ArrayImpl() = default;

        ArrayImpl(const ArrayImpl &) = delete;

        ArrayImpl &operator=(const ArrayImpl &) = delete;

        [[nodiscard]] void *get_raw_buffer() const {
            if (buffer_container == nullptr) throw std::runtime_error("data buffer is null");
            return static_cast<char *>(buffer()) + offset;
        }

        [[nodiscard]] size_t get_column_stride() const {
            if (buffer_container == nullptr) throw std::runtime_error("data buffer is null");
            return buffer_container->get_true_column_len();
        }

        void multiply(const ArrayImpl *other,
                      ArrayImpl *dest,
                      const std::vector<size_t> &common_shape,
                      const std::vector<size_t> &this_stride,
                      const std::vector<size_t> &other_stride) const {

            this->m_dispatcher->hadamard_product(
                get_raw_buffer(),
                other->get_raw_buffer(),
                dest->get_raw_buffer(),
                common_shape.data(),
                common_shape.size(),
                this_stride.data(),
                other_stride.data(),
                dest->get_column_stride(),
                dtype);
        }

        void add(const ArrayImpl *other,
                 ArrayImpl *dest,
                 const std::vector<size_t> &common_shape,
                 const std::vector<size_t> &this_stride,
                 const std::vector<size_t> &other_stride) const {

            this->m_dispatcher->element_wise_add(
            get_raw_buffer(),
            other->get_raw_buffer(),
            dest->get_raw_buffer(),
                common_shape.data(),
                common_shape.size(),
                this_stride.data(),
                other_stride.data(),
                dest->get_column_stride(),
                dtype);
        }

        void permute(ArrayImpl *dest, const std::vector<size_t> &permute_mask) const {
            m_dispatcher->permute(
                get_raw_buffer(),
                dest->get_raw_buffer(),
                shape.size(),
                shape.data(),
                permute_mask.data(),
                stride.data(),
                dest->stride.data(),
                dtype);
        }

        void gemv(
            const ArrayImpl *matrix,
            const ArrayImpl *vector,
            const void *alpha,
            const void *beta) {

            m_dispatcher->gemv(
                matrix->get_raw_buffer(),
                vector->get_raw_buffer(),
                get_raw_buffer(),
                alpha,
                beta,
                matrix->shape[0],
                matrix->shape[1],
                matrix->stride[0],
                dtype);
        }


        [[nodiscard]] bool requires_grad() const {
            return node != nullptr;
        }

        [[nodiscard]] std::string generate_description() const {
            std::stringstream ss;
            ss << "############## Details ##############\n";
            ss << "Shape: ";
            shape_to_str(shape, ss);
            ss << "\n";
            ss << "Device: " << device_to_string(device) << "\n";
            ss << "Dtype: " << dtype_to_string(dtype) << "\n";
            ss << "#####################################\n";

            return ss.str();
        }

        void retain_grad(const bool state) {
            if (!requires_grad()) throw std::runtime_error("brarray must require gradients in order to retain them");
            if (!node->retain_grad && state) node->retain_grad = true;
            else if(node->retain_grad && !state) node->retain_grad = false;
        }

        void elementwise_link_nodes(const ArrayImpl *input, const ArrayImpl *other, const std::shared_ptr<AutogradNode> &node) {
            if (!requires_grad()) throw std::runtime_error("brarray does not require gradients");
            if (input->requires_grad()) input->node->add_next_node(node, 0);
            if (other->requires_grad()) other->node->add_next_node(node, 1);
            this->node->prev_nodes.push_back(node);
            node->add_next_node(this->node, 0);
        }
    };

    Brarray::Brarray(Device const device, Dtype const dtype, std::vector<size_t> const &shape): impl(std::make_unique<ArrayImpl>(device, dtype, shape)) {}

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
        return impl->requires_grad();
    }

    void Brarray::retain_grad(const bool state) {
        impl->retain_grad(state);
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

    template<typename Type>
    const void * get_vector_ptr(const Dtype dtype, std::vector<size_t> const &shape, const std::vector<Type> &data) {
        is_invalid(dtype);

        const Dtype provided{get_dtype_from_type<Type>::type};

        if (dtype != provided)
            throw std::runtime_error(
                "Template dtype T does not match brarray dtypes");

        if (shape.empty()) throw std::runtime_error("an empty shape was provided");
        if (data.empty()) throw std::runtime_error("an empty vector was provided");

        const size_t total_elements{shape[shape.size() - 1] * calculate_total_rows(shape)};
        if (total_elements != data.size()) throw std::runtime_error("given shape cannot compensate provided data");

        return data.data();
    }

    template<typename Type>
    Brarray::Brarray(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<Type> &data):
        impl(std::make_unique<ArrayImpl>(device, dtype, device_is_host(device) ? HOST_TO_HOST : HOST_TO_DEVICE,
            shape, shape[shape.size() - 1], get_vector_ptr<Type>(dtype, shape, data))){}

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

    Brarray::~Brarray() = default;

    Brarray::Brarray(): impl(std::make_unique<ArrayImpl>()) {}

    Brarray::Brarray(const Brarray &other) : impl(std::make_unique<ArrayImpl>()) {
        impl->shape = other.impl->shape;
        impl->stride = other.impl->stride;
        impl->device = other.impl->device;
        impl->dtype = other.impl->dtype;
        impl->m_dispatcher = other.impl->m_dispatcher;
        impl->total_rows = other.impl->total_rows;
        impl->buffer_container = std::make_shared<BufferContainer>(
            other.impl->buffer_container.get(),
            other.impl->offset,
            other.impl->shape);
    }

    Brarray &Brarray::operator=(const Brarray &other) {
        if (this == &other)
            return *this;

        impl->shape = other.impl->shape;
        impl->stride = other.impl->stride;
        impl->device = other.impl->device;
        impl->dtype = other.impl->dtype;
        impl->m_dispatcher = other.impl->m_dispatcher;
        impl->total_rows = other.impl->total_rows;
        impl->buffer_container = std::make_shared<BufferContainer>(
            other.impl->buffer_container.get(),
            other.impl->offset,
            other.impl->shape);

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

        const auto [shape, stride_one, stride_two]{
            validate_element_wise(
                input,
                other,
                input.impl->buffer_container->get_true_column_len(),
                other.impl->buffer_container->get_true_column_len())};

        Brarray result(input.get_device(), input.get_dtype(), shape);

        if (track_gradients && (input.requires_grad() || other.requires_grad())) {
            result.requires_grad(true);
            result.retain_grad(false);
            const auto mlt{std::make_shared<Multiply>(input, other, input.impl->node, other.impl->node)};
            result.impl->elementwise_link_nodes(input.impl.get(), other.impl.get(), mlt);
        }

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
                input.impl->buffer_container->get_true_column_len(),
                other.impl->buffer_container->get_true_column_len())};

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
                other.impl->buffer_container->get_true_column_len())};

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
                other.impl->buffer_container->get_true_column_len())};

        input.impl->add(
            other.impl.get(),
            input.impl.get(),
            shape,
            stride_one,
            stride_two);
    }

    template<typename T>
    void gemv(
    Brarray &result,
    const Brarray &matrix,
    const Brarray &vector,
    T alpha,
    T beta) {
        constexpr Dtype dtype{get_dtype_from_type<T>::type};
        is_invalid(dtype);

        if (dtype != matrix.get_dtype() || dtype != result.get_dtype() || dtype != vector.get_dtype())
            throw std::runtime_error("Template dtype T does not match brarray dtypes");

        if (!vector.is_vector()) throw std::runtime_error("supplied 'vector' is not a vector");
        if (!result.is_vector()) throw std::runtime_error("the current brarray is not a vector");
        if (!matrix.is_matrix()) throw std::runtime_error("supplied 'matrix' is not a matrix");

        if (matrix.get_shape()[1] != vector.get_shape()[0])
            throw std::runtime_error(
                "vector and matrix have different columns lengths");

        if (matrix.get_shape()[0] != result.get_shape()[0])
            throw std::runtime_error(
                "the current brarray must be of shape (matrix.get_shape()[0])");

        if (matrix.get_device() != vector.get_device() || result.get_device() != matrix.get_device()) {
            throw std::runtime_error("vector, matrix and the current brarray are not on the same device");
        }

        if (matrix.get_dtype() != vector.get_dtype() || matrix.get_dtype() != result.get_dtype()) {
            throw std::runtime_error("vector, matrix and current brarray have different dtypes");
        }

        result.impl->gemv(matrix.impl.get(), vector.impl.get(), &alpha, &beta);
    }

    template<typename T>
    Brarray gemv(
        const Brarray &matrix,
        const Brarray &vector,
        T alpha,
        T beta) {
        Brarray result(matrix.get_device(), matrix.get_dtype(), {matrix.get_shape()[0]});
        gemv<T>(result, matrix, vector, alpha, beta);
        return result;
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

    std::vector<size_t> get_permuted_shape(const std::vector<size_t> &shape, const std::vector<size_t> &mask) {
        if (mask.size() != shape.size()) throw std::runtime_error("permutation is missing dimensions");

        std::vector<size_t> ret;
        ret.reserve(shape.size());

        // arithmetic series sum
        size_t sm = (shape.size() - 1) * shape.size() / 2;
        for (const auto &dim: mask) {
            if (dim >= shape.size()) throw std::runtime_error("invalid permutation provided"); //TODO make more descriptive
            ret.push_back(shape[dim]);
            sm -= dim;
        }

        if (sm != 0) throw std::runtime_error("invalid permutation provided");
        return ret;
    }

    Brarray Brarray::permute(const std::vector<size_t>& dims, const bool requires_grad) const {
        is_invalid(this->get_dtype());
        if (is_vector()) throw std::runtime_error("cannot permute brarray with only 1 dimension");
        const std::vector permuted_shape{get_permuted_shape(impl->shape, dims)};

        if (permuted_shape == impl->shape) {
            Brarray ret{*this};
            return ret;
        }

        Brarray ret(this->get_device(), this->get_dtype(), permuted_shape);
        if (requires_grad) {} // TODO add gradient integration
        impl->permute(ret.impl.get(), dims);
        return ret;
    }


    bool Brarray::is_matrix() const {
        return get_shape().size() == 2;
    }

    bool Brarray::is_vector() const {
        return impl->total_rows == 1;
    }

    bool Brarray::is_scalar_equivalent() const {
        if (is_vector()) return get_shape()[0] == 1;
        return false;
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
                scalar.impl->get_raw_buffer(),
                static_cast<char *>(impl->get_raw_buffer()) + index * dtype_to_bytes(this->impl->dtype),
                dtype_to_bytes(this->impl->dtype),
                device_is_host(get_device()) ? HOST_TO_HOST : DEVICE_TO_DEVICE);

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
        ret.impl->total_rows = calculate_total_rows(ret.get_shape());

        return ret;
    }

    /**
     * provides access to the underlying buffer
     * @tparam T the type that the ptr should be cast too, it must match the Dtype
     * @return the raw ptr buffer
     */
    template<typename T>
    T * Brarray::get_buffer() const{
        const Dtype current{this->get_dtype()};
        if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
            throw std::runtime_error(
                "provided buffer type does not match array type: " + dtype_to_string(current));
        }

        return static_cast<T *>(impl->get_raw_buffer());
    }

    template<typename Type>
    Type Brarray::item() const{
        const Dtype current{this->get_dtype()};
        if (constexpr Dtype given = get_dtype_from_type<Type>::type; given != current) {
            throw std::runtime_error(
                "provided type does not match array type: " + dtype_to_string(current));
        }

        if (!this->is_scalar_equivalent()) {
            throw std::out_of_range("array can only return item if it contains a single element");
        }

        return *get_buffer<Type>();
    }

    template<typename T>
    void Brarray::set_item(T value) {
        if (!this->is_scalar_equivalent()) {
            throw std::out_of_range("cannot set singular value to non scalar");
        }

        get_buffer<T>()[0] = value;
    }

    template<typename Type>
    void print_vector(const Type *data_ptr, const size_t length, std::stringstream &ss) {
        ss << "[";
        for (size_t i{0}; i < length - 1; ++i) ss << data_ptr[i] << ", ";
        ss << data_ptr[length - 1] << "]";
    }

    void print_vector(const int8_t *data_ptr, const size_t length, std::stringstream &ss) {
        ss << "[";
        for (size_t i{0}; i < length - 1; ++i) ss << static_cast<int>(data_ptr[i]) << ", ";
        ss << static_cast<int>(data_ptr[length - 1]) << "]";
    }

    void to_string_helper(std::stringstream &ss, const Brarray &br, const std::string &gap) {
        if (br.is_vector()) {
            ss << gap;
            switch (br.get_dtype()) {
                case INT8: {
                    print_vector(br.get_buffer<int8_t>(), br.get_shape()[0], ss);
                    break;
                }
                case INT16: {
                    print_vector(br.get_buffer<int16_t>(), br.get_shape()[0], ss);
                    break;
                }
                case INT32: {
                    print_vector(br.get_buffer<int32_t>(), br.get_shape()[0], ss);
                    break;
                }
                case INT64: {
                    print_vector(br.get_buffer<int64_t>(), br.get_shape()[0], ss);
                    break;
                }
                case FLOAT32: {
                    print_vector(br.get_buffer<float>(), br.get_shape()[0], ss);
                    break;
                }
                case FLOAT64: {
                    print_vector(br.get_buffer<double>(), br.get_shape()[0], ss);
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
        outs << b.impl->generate_description();
        outs << str;
        return outs;
    }

    INSTANTIATE_OPERATOR(Brarray::operator*);
    INSTANTIATE_OPERATOR(Brarray::operator+);
    INSTANTIATE_INPLACE_OPERATOR(imult);
    INSTANTIATE_INPLACE_OPERATOR(iadd);
    INSTANTIATE_VECTOR_CONSTRUCTOR();
    INSTANTIATE_GET_BUFFER();
    INSTANTIATE_SET_ITEM();
    INSTANTIATE_GET_ITEM();
    INSTANTIATE_GEMV_WHOLE();
    INSTANTIATE_GEMV_PARTIAL();
}
