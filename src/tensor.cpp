//
// Created by sriram on 2/8/25.
//

#include "tensor.h"
#include <iomanip>
#include "enums.h"

static size_t product(const std::vector<size_t> &vec) {
    size_t prod{1};
    for (size_t const &data: vec)
        prod *= data;

    return prod;
}

namespace cobraml::core {
    Tensor::Tensor(): shape({}) {}

    Tensor::Tensor(std::vector<size_t> shape, Device const device, Dtype const dtype):
        Barray(product(shape), device, dtype), shape(std::move(shape)) {

        if (this->shape.size() == 1) {
            this->shape = std::move(std::vector{1, this->shape[0]});
        }
    }

    Tensor::Tensor(Tensor const &other): Barray(other), shape(other.shape) {}

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this == &other) {
            return *this;
        }

        Barray::operator=(other);
        this->shape = other.shape;
        return *this;
    }

    Tensor::~Tensor() = default;

    const std::vector<size_t> &Tensor::get_shape() const {
        return shape;
    }

    bool Tensor::is_matrix() const {
        return this->shape.size() == 2;
    }

    Matrix Tensor::to_matrix() const {
        if (!is_matrix()) {
            throw std::runtime_error("tensor has too many dimensions to be considered a matrix");
        }

        Matrix ret(*this);

        ret.rows = this->shape[0];
        ret.columns = this->shape[1];

        return ret;
    }

    Tensor Tensor::operator[](size_t const index) const {
        if (this->shape.empty()) {
            throw std::runtime_error("tensor is completely empty");
        }

        if (index >= this->shape[0]) {
            throw std::out_of_range("index is out of range");
        }

        Tensor ret{*this};

        if (this->shape.size() == 1) {
            ret.increment_offset(index);
            ret.set_length(1);
            ret.shape = std::move(std::vector<size_t>{1, 1});
            return ret;
        }

        std::vector new_shape(ret.shape.begin() + 1, ret.shape.end());
        ret.shape = std::move(new_shape);

        size_t const skip_amt{product(ret.shape)};
        ret.increment_offset(index * skip_amt);
        ret.set_length(skip_amt);
        return ret;
    }

}
