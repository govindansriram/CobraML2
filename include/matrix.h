//
// Created by sriram on 12/15/24.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include "enums.h"
#include "iostream"
#include "barray.h"

/**
 * TODO:
 * Test Shape
 * Test Indexing
 * Test GEMV with invalid alpha and beta
 * Test and Update To Tensor
 * Create Tensor class
 */
namespace cobraml::core {

    class Matrix final : public Array{
        size_t rows;
        size_t columns;

        // private constructor for stuff such as tensor to Matrix
        Matrix(Array const &other);

        friend class Tensor;
    public:
        struct Shape {
            size_t rows;
            size_t columns;
            bool operator==(const Shape& other) const;
        };

        /**
         * constructor that creates a zerod matrix of shape (rows, columns)
         * @param rows the # of rows in the matrix
         * @param columns the # of columns in the matrix
         * @param device the device of the matrix being constructed
         * @param dtype the dtype of the matrix being constructed
         */
        Matrix(size_t rows, size_t columns, Device device, Dtype dtype);

        Matrix();
        Matrix(Matrix const &other);
        Matrix& operator=(const Matrix& other);
        Matrix operator[] (size_t index) const;

        /**
         * @return True if matrix qualifies as a vector
         */
        [[nodiscard]] bool is_vector() const;

        /**
         * @return True if matrix qualifies as a scalar
         */
        [[nodiscard]] bool is_scalar() const;

        /**
        * @return the shape of the matrix
        */
        [[nodiscard]] Shape get_shape() const;

        /**
         * prints the contents of the matrix in tabular format
         * @param show_description
         */
        void print(bool show_description) const override;

        ~Matrix() override;

        // Start of the Friend API

        /**
         * Generalized Matrix Vector Multiplication.
         * Performs y=αAx+βy
         *
         * @param matrix A
         * @param vector x
         * @param result y
         * @param alpha α
         * @param beta β
         */
        template<typename T>
        friend void gemv(const Matrix &matrix, const Matrix &vector, Matrix &result, T alpha, T beta);

        template<typename T>
        friend Matrix from_vector(const std::vector<std::vector<T>> &mat, Device device);
    };

    template<typename T>
    Matrix from_vector(const std::vector<std::vector<T>> &mat, Device const device) {
        constexpr Dtype dtype{get_dtype_from_type<T>::type};
        is_invalid(dtype);

        const size_t rows{mat.size()};

        if (!rows) {
            throw std::runtime_error("vector is empty");
        }

        const size_t columns{mat[0].size()};
        Matrix ret(rows, columns, device, dtype);

        if (rows == 1) {
            ret.copy_vector(mat[0]);
            return ret;
        }

        size_t count{0};

        for (const std::vector<T> &row: mat) {
            if (row.size() != columns) {
                throw std::runtime_error("matrix is not rectangular");
            }

            ret[count].copy_vector(row);
            ++count;
        }

        return ret;
    }

    template<typename T>
    void gemv(const Matrix &matrix, const Matrix &vector, Matrix &result, const T alpha, const T beta) {
        if (!vector.is_vector()) {
            throw std::runtime_error("vector is a matrix");
        }

        if (!result.is_vector()) {
            throw std::runtime_error("result is a matrix");
        }

        if (matrix.columns != vector.columns) {
            throw std::runtime_error("vector and matrix have different columns lengths");
        }

        if (matrix.rows != result.columns) {
            throw std::runtime_error("result must be size 1, rows(matrix)");
        }

        if (matrix.get_device() != vector.get_device() || matrix.get_device() != result.get_device()) {
            throw std::runtime_error("vector, matrix and result are not on the same device");
        }

        if (matrix.get_dtype() != vector.get_dtype() || matrix.get_dtype() != result.get_dtype()) {
            throw std::runtime_error("vector, matrix and result share different dtypes");
        }

        const Dtype current{matrix.get_dtype()};
        if (constexpr Dtype given = get_dtype_from_type<T>::type; given != current) {
            throw std::runtime_error(
                "alpha and beta has a invalid dtype, expected " + dtype_to_string(current));
        }

        result.gemv(matrix, vector, matrix.rows, matrix.columns, &alpha, &beta);
    }
}

#endif //MATRIX_H
