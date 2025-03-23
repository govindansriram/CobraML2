//
// Created by sriram on 1/25/25.
//

#ifndef BARRAY_H
#define BARRAY_H
#include <memory>
#include <queue>
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

        /**
         * get the raw pointer backing an array, do not free this pointer
         *
         * @return a raw void pointer to the buffer data
         */
        [[nodiscard]] void *get_raw_buffer() const;

        /**
         * updates the array buffer of a vector with the reassigned values
         * @param source the replacement data
         */
        void reassign_vector(const void *source);

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
         * @param rows
         * @param columns
         * @param alpha α
         * @param beta β
         */
        void gemv(
            const Brarray &matrix,
            const Brarray &vector,
            size_t rows,
            size_t columns,
            const void *alpha,
            const void *beta);

        [[nodiscard]] virtual std::string generate_description() const;

        Brarray(Device device, Dtype dtype, std::vector<size_t> const &shape, const void *ptr);

        static const void *validated_get_data(
            Dtype dtype_vec,
            Dtype provided,
            size_t shape,
            std::vector<size_t> const &provided_shape,
            const void * data);

    public:
        Brarray(Device device, Dtype dtype, std::vector<size_t> const &shape);

        virtual ~Brarray();

        Brarray();

        Brarray(const Brarray &other);

        Brarray &operator=(const Brarray &other);

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


        /**
         * provides access to the underlying buffer
         * @tparam T the type that the ptr should be cast too, it must match the Dtype
         * @return the raw ptr buffer
         */
        template<typename T>
        const T *get_buffer() const{
            const Dtype current{this->get_dtype()};
            if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
                throw std::runtime_error(
                    "provided buffer type does not match array type: " + dtype_to_string(current));
            }

            return static_cast<T *>(this->get_raw_buffer());
        }

        template<typename T, std::enable_if<std::is_arithmetic_v<T> >* = nullptr>
        Brarray(const Device device, const Dtype dtype, std::vector<size_t> const &shape,
                const std::vector<T> &data): Brarray(device, dtype, shape, validated_get_data(
                                                         get_dtype_from_type<T>::type,
                                                         dtype,
                                                         data.size(),
                                                         shape,
                                                         data.data())) {}

        template<typename T>
        friend void gemv(
            Brarray &result,
            const Brarray &matrix,
            const Brarray &vector,
            size_t rows,
            size_t columns,
            T alpha,
            T beta);

        template<typename T>
        friend Brarray gemv(
            const Brarray &matrix,
            const Brarray &vector,
            size_t rows,
            size_t columns,
            T alpha,
            T beta);

        friend std::ostream &operator<<(std::ostream &outs, const Brarray &b);

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
        void set_item(T value) const {
            const Dtype current{this->get_dtype()};
            if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
                throw std::runtime_error(
                    "provided buffer type does not match array type: " + dtype_to_string(current));
            }

            if (!this->is_scalar_equivalent()) {
                throw std::out_of_range("cannot set singular value to non scalar");
            }

            reassign_vector(&value);
        }
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
        size_t const rows,
        size_t const columns,
        T alpha,
        T beta) {
        constexpr Dtype dtype{get_dtype_from_type<T>::type};
        is_invalid(dtype);

        if (dtype != matrix.get_dtype() || dtype != result.get_dtype() || dtype != vector.get_dtype())
            throw std::runtime_error("Template dtype T does not match brarray dtypes");

        result.gemv(matrix, vector, rows, columns, &alpha, &beta);
    }

    template<typename T>
    Brarray gemv(
        const Brarray &matrix,
        const Brarray &vector,
        size_t const rows,
        size_t const columns,
        T alpha,
        T beta) {
        Brarray result(matrix.get_device(), matrix.get_dtype(), {matrix.get_shape()[0]});
        gemv(result, matrix, vector, rows, columns, alpha, beta);
        return result;
    }
}

#endif //BARRAY_H
