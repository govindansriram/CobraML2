//
// Created by sriram on 12/19/24.
//


#include "matrix.h"
#include <iomanip>
#include <iostream>
#include "enums.h"

/**
 * TODO: double check all matrix functions, then add print, then start tensor functions.
 */
namespace cobraml::core {

    Matrix::Matrix(size_t const rows, size_t const columns, Device const device, Dtype const dtype):
        Array(rows * columns, device, dtype),
        rows(rows),
        columns(columns){
        is_invalid(dtype);
    }

    Matrix::Shape Matrix::get_shape() const {
        Shape sh{};
        sh.columns = columns;
        sh.rows = rows;
        return sh;
    }

    Matrix::Matrix(Array const &other): Array(other), rows(0), columns(0) {}
    Matrix::Matrix(Matrix const &other): Array(other), rows(other.rows), columns(other.columns) {}


    Matrix::~Matrix() = default;

    bool Matrix::Shape::operator==(const Shape &other) const {
        return this->rows == other.rows && this->columns == other.columns;
    }

    bool Matrix::is_vector() const {
        return rows == 1;
    }

    bool Matrix::is_scalar() const {
        return this->is_vector() && columns == 1;
    }

    Matrix Matrix::operator[](size_t const index) const{

        Matrix ret = *this;
        ret.rows = 1;

        if (this->is_vector()) {
            if (index >= this->columns) {
                throw std::out_of_range("index is out of range");
            }

            ret.columns = 1;
            ret.increment_offset(index);

            ret.set_length(1);

            return ret;
        }

        if (index >= this->get_shape().rows) {
            throw std::out_of_range("index is out of range");
        }

        ret.columns = this->get_shape().columns;

        ret.increment_offset(this->columns * index);
        ret.set_length(columns);

        return ret;
    }

    Matrix::Matrix(): rows(0), columns(0) {}

    void print_num(void *buffer, Dtype const dtype) {
        switch (dtype) {
            case INT8: {
                const auto num = *static_cast<char *>(buffer);
                std::cout << static_cast<int>(num);
                return;
            }
            case INT16: {
                std::cout << *static_cast<int16_t *>(buffer);
                return;
            }
            case INT32: {
                std::cout << *static_cast<int32_t *>(buffer);
                return;
            }
            case INT64: {
                std::cout << *static_cast<int64_t *>(buffer);
                return;
            }
            case FLOAT32: {
                std::cout << *static_cast<float *>(buffer);
                return;
            }
            case FLOAT64: {
                std::cout << *static_cast<double *>(buffer);
                return;
            }
            case INVALID: {
                is_invalid(dtype);
            }
        }
    }

    Matrix& Matrix::operator=(const Matrix &other) {
        if (this == &other) {
            return *this;
        }

        Array::operator=(other);
        this->rows = other.rows;
        this->columns = other.columns;

        return *this;
    }

    void print_description(Matrix const * mat) {
        auto [rows, columns]{mat->get_shape()};
        std::cout << "############## Details ##############\n";
        std::cout << "Shape: " << "(" << rows << ", " << columns << ")" << '\n';
        std::cout << "Device: " << device_to_string(mat->get_device()) << '\n';
        std::cout << "Dtype: " << dtype_to_string(mat->get_dtype()) << '\n';
        std::cout << "#####################################\n";
    }

    void Matrix::print(bool const show_description) const {
        if (this->get_dtype() == INVALID) {
            throw std::runtime_error("cannot print matrix with invalid dtype");
        }

        if (show_description)
            print_description(this);

        for (size_t i = 0; i < this->get_shape().rows; ++i) {
            Array vec{(*this)[i]};
            vec.print(false);
        }
    }
}
