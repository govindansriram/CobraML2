//
// Created by sriram on 1/25/25.
//

#ifndef BARRAY_H
#define BARRAY_H
#include <memory>
#include <sstream>
#include <vector>
#include "enums.h"

namespace cobraml::core {
#define PRINT_LIMIT 30

    /**
     * The Cobraml Array
     */
    class Brarray {
    protected:
        struct ArrayImpl;
        std::unique_ptr<ArrayImpl> impl;

        [[nodiscard]] bool is_vector() const;

        [[nodiscard]] bool is_scalar_equivalent() const;

        [[nodiscard]] bool is_matrix() const;

        // Mathematical operations

        /**
         * Generalized Matrix Vector Multiplication.
         * Performs y=αAx+βy
         *
         * @param matrix A
         * @param vector x
         * @param alpha α
         * @param beta β
         */
        void gemv(
            const Brarray &matrix,
            const Brarray &vector,
            const void *alpha,
            const void *beta);

        [[nodiscard]] virtual std::string generate_description() const;

        void to_string_helper(std::stringstream & ss, const Brarray & br, const std::string & gap) const;

    public:
        Brarray(Device device, Dtype dtype, std::vector<size_t> const &shape);

        virtual ~Brarray();

        Brarray();

        Brarray(const Brarray &other);

        Brarray &operator=(const Brarray &other);

        Brarray(Brarray &&other) noexcept;
        Brarray& operator=(Brarray&& other) noexcept;

        [[nodiscard]] virtual std::string to_string() const;

        /**
         * @return the dtype of the brarray
         */
        [[nodiscard]] Dtype get_dtype() const;

        /**
        * @return the Device of the brarray
        */
        [[nodiscard]] Device get_device() const;

        /**
         * @return the Device of the brarray
         */
        [[nodiscard]] std::vector<size_t> get_shape() const;

        /**
         * @return get the stride of the brarray
         */
        [[nodiscard]] std::vector<size_t> get_stride() const;


        // Gradient Descent
        void requires_grad(bool state);
        void retain_grad(bool state);
        [[nodiscard]] bool retain_grad() const;
        [[nodiscard]] bool requires_grad() const;
        Brarray& get_gradient();
        void backwards();


        [[nodiscard]] Brarray shared_copy() const;
        [[nodiscard]] Brarray permute(const std::vector<size_t>& dims, bool requires_grad=true) const;

        /**
         * Hadamard Product (Element WIse Multiplication)
         * @param other the Multiplier
         * @return the element wise product
         */
        Brarray operator*(const Brarray & other) const;
        Brarray operator+(const Brarray & other) const;

        template<typename Dtype>
        Brarray operator*(Dtype other) const;

        template<typename Dtype>
        Brarray operator+(Dtype other) const;
        // Brarray operator-(const Brarray & other) const;


        /**
         * provides access to the underlying buffer
         * @tparam T the type that the ptr should be cast too, it must match the Dtype
         * @return the raw ptr buffer
         */
        template<typename T>
        T *get_buffer() const;

        template<typename Type>
        Brarray(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<Type> &data);

        // Start of the Friend API
        template<typename T>
        friend void gemv(
            Brarray &result,
            const Brarray &matrix,
            const Brarray &vector,
            T alpha,
            T beta);

        template<typename T>
        friend Brarray gemv(
            const Brarray &matrix,
            const Brarray &vector,
            T alpha,
            T beta);

        friend Brarray mult(const Brarray &input, const Brarray &other, bool track_gradients);
        friend Brarray add(const Brarray &input, const Brarray &other, bool track_gradients);

        template<typename Dtype>
        friend void imult(Brarray &input, Dtype other);
        friend void imult(Brarray &input, const Brarray &other);

        template<typename Dtype>
        friend void iadd(Brarray &input, Dtype other);
        friend void iadd(Brarray &input, const Brarray &other);

        friend Brarray pow(const Brarray &brarray, const Brarray &exponent, bool track_gradients);
        friend void ipow(const Brarray &brarray, const Brarray &exponent);

        friend void iadd(const Brarray &brarray_one, const Brarray &brarray_two, bool track_gradients);

        friend Brarray sub(const Brarray &brarray_one, const Brarray &brarray_two, bool track_gradients);
        friend void isub(const Brarray &brarray_one, const Brarray &brarray_two, bool track_gradients);

        friend std::ostream &operator<<(std::ostream &outs, const Brarray &b);

        // End of the Friend API

        /**
         * Gets the tensor present at that specific dimension shares data with the original tensor
         * @param index the tensor to copy at that specific index
         * @return the tensor at that index
         */
        Brarray operator[](size_t index) const;

        template<typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
        T item() const{
            const Dtype current{this->get_dtype()};
            if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
                throw std::runtime_error(
                    "provided type does not match array type: " + dtype_to_string(current));
            }

            if (!this->is_scalar_equivalent()) {
                throw std::out_of_range("array can only return item if it contains a single element");
            }

            return *get_buffer<T>();
        }

        template<typename T>
        void set_item(T value);
    };

    template<typename T, std::enable_if_t<!std::is_same_v<int8_t, T> && std::is_arithmetic_v<T>>* = nullptr>
    void print_vector(void *data_ptr, const size_t length, std::stringstream &ss) {
        ss << "[";
        auto ptr{static_cast<T *>(data_ptr)};
        for (size_t i{0}; i < length - 1; ++i) ss << ptr[i] << ", ";
        ss << ptr[length - 1] << "]";
    }

    template<typename T, std::enable_if_t<std::is_same_v<int8_t, T> >* = nullptr>
    void print_vector(void *data_ptr, const size_t length, std::stringstream &ss) {
        ss << "[";
        const auto ptr{static_cast<int8_t *>(data_ptr)};
        for (size_t i{0}; i < length - 1; ++i) ss << static_cast<int>(ptr[i]) << ", ";
        ss << static_cast<int>(ptr[length - 1]) << "]";
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

        result.gemv(matrix, vector, &alpha, &beta);
    }

    template<typename T>
    Brarray gemv(
        const Brarray &matrix,
        const Brarray &vector,
        T alpha,
        T beta) {
        Brarray result(matrix.get_device(), matrix.get_dtype(), {matrix.get_shape()[0]});
        gemv(result, matrix, vector, alpha, beta);
        return result;
    }

    Brarray mult(const Brarray &input, const Brarray &other, bool track_gradients=true);
    void imult(Brarray &input, const Brarray &other);
    template<typename Dtype>
    void imult(Brarray &input, Dtype other);

    Brarray add(const Brarray &input, const Brarray &other, bool track_gradients);
    void iadd(Brarray &input, const Brarray &other);
    template<typename Dtype>
    void iadd(Brarray &input, Dtype other);

    // Brarray pow(const Brarray &brarray, const Brarray &exponent, bool track_gradients=true);
    // Brarray sub(const Brarray &brarray_one, const Brarray &brarray_two, bool track_gradients=true);
}

#endif //BARRAY_H
