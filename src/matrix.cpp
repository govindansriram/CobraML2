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
        Barray(rows * columns, device, dtype),
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

    Matrix::Matrix(Barray const &other): Barray(other), rows(0), columns(0) {}
    Matrix::Matrix(Matrix const &other): Barray(other), rows(other.rows), columns(other.columns) {}


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

    Matrix& Matrix::operator=(const Matrix &other) {
        if (this == &other) {
            return *this;
        }

        Barray::operator=(other);
        this->rows = other.rows;
        this->columns = other.columns;

        return *this;
    }

     std::string Matrix::generate_description() const {
        std::stringstream ss;
        auto [rows, columns]{this->get_shape()};
        ss << "############## Details ##############\n";
        ss << "Shape: " << "(" << rows << ", " << columns << ")" << '\n';
        ss << "Device: " << device_to_string(this->get_device()) << '\n';
        ss << "Dtype: " << dtype_to_string(this->get_dtype()) << '\n';
        ss << "#####################################\n";

        return ss.str();
    }

    std::string Matrix::to_string(int8_t const gap) const {
        std::stringstream spaces;

        for (int8_t i{0}; i < gap; ++i) {
            spaces << " ";
        }

        auto const sp1_str = spaces.str();

        std::stringstream obj_str;
        obj_str << sp1_str << "[\n";

        if (rows == 1) {
            Barray vec{(*this)};
            obj_str << vec.to_string(static_cast<int8_t>(gap + 4));
            obj_str << sp1_str << "]\n";
            return obj_str.str();
        }

        size_t stop{this->rows / 2};
        size_t start_1{stop};
        bool print_dots{};

        if (this->get_shape().rows > PRINT_LIMIT) {
            print_dots = true;
            stop = 3;
            start_1 = this->rows - 3;
        }

        for (size_t i = 0; i < stop; ++i) {
            Barray vec{(*this)[i]};
            obj_str << vec.to_string(static_cast<int8_t>(gap + 4));
        }

        if (print_dots)
            obj_str << sp1_str << "    " << "..." << std::endl;

        for (; start_1 < this->rows; ++start_1) {
            Barray vec{(*this)[start_1]};
            obj_str << vec.to_string(static_cast<int8_t>(gap + 4));
        }

        obj_str << sp1_str << "]\n";

        return obj_str.str();
    }
}
