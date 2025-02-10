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

static std::string generate_spaces(int8_t const gap) {
    std::stringstream spaces;

    for (int8_t i{0}; i < gap; ++i) {
        spaces << " ";
    }

    return spaces.str();
}

namespace cobraml::core {
    Tensor::Tensor(): shape({}) {
    }

    Tensor::Tensor(std::vector<size_t> shape, Device const device, Dtype const dtype): Barray(product(shape), device,
        dtype), shape(std::move(shape)) {
        if (this->shape.size() == 1) {
            this->shape = std::move(std::vector{1, this->shape[0]});
        }
    }

    Tensor::Tensor(Tensor const &other): Barray(other), shape(other.shape) {
    }

    Tensor &Tensor::operator=(const Tensor &other) {
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

        if (this->shape[0] != 1 && index >= this->shape[0]) {
            throw std::out_of_range("index is out of range");
        }

        if (this->shape[0] == 1 && index >= this->shape[1]) {
            throw std::out_of_range("index is out of range");
        }

        Tensor ret{*this};
        std::vector<size_t> new_shape;

        if (shape.size() == 2 && shape[0] != 1) {
            new_shape.reserve(2);
            new_shape.push_back(1);
            new_shape.push_back(this->shape[1]);
        } else if (shape.size() == 2) {
            new_shape.reserve(2);
            new_shape.push_back(1);
            new_shape.push_back(1);
        } else {
            new_shape = std::vector(ret.shape.begin() + 1, ret.shape.end());
        }

        ret.shape = std::move(new_shape);
        size_t const skip_amt{product(ret.shape)};
        ret.increment_offset(index * skip_amt);
        ret.set_length(skip_amt);
        return ret;
    }

    std::string Tensor::generate_description() const {
        std::stringstream ss;
        ss << "############## Details ##############\n";
        ss << "Shape: (";
        size_t i = 0;
        for (; i < shape.size() - 1; ++i) {
            ss << shape[i];
            ss << ", ";
        }
        ss << shape[i];
        ss << ")\n";
        ss << "Device: " << device_to_string(this->get_device()) << "\n";
        ss << "Dytpe: " << dtype_to_string(this->get_dtype()) << "\n";
        ss << "#####################################\n";

        return ss.str();
    }


    void print_helper(std::stringstream &stream, Tensor const &tens, int8_t const gap) {
        if (tens.get_shape().size() == 2) {
            stream << tens.to_matrix().to_string(gap);
            return;
        }

        auto const space_string = generate_spaces(gap);

        size_t stop{tens.get_shape()[0] / 2};
        size_t start_1{stop};
        bool print_dots{};

        if (tens.get_shape()[0] > PRINT_LIMIT) {
            print_dots = true;
            stop = 3;
            start_1 = tens.get_shape()[0] - 3;
        }

        for (size_t i = 0; i < stop; ++i) {
            if (tens.get_shape().size() != 3) {
                stream << space_string << "[\n";
                print_helper(stream, tens[i], static_cast<int8_t>(gap + 4));
            } else {
                print_helper(stream, tens[i], gap);
            }
            if (tens.get_shape().size() != 3) {
                stream << space_string << "]\n";
            }
        }

        if (print_dots) {
            stream << space_string << "..." << std::endl;
        }

        for (; start_1 < tens.get_shape()[0]; ++start_1) {
            if (tens.get_shape().size() != 3) {
                stream << space_string << "[\n";
                print_helper(stream, tens[start_1], static_cast<int8_t>(gap + 4));
            } else {
                print_helper(stream, tens[start_1], gap);
            }
            if (tens.get_shape().size() != 3) {
                stream << space_string << "]\n";
            }
        }
    }

    std::string Tensor::to_string(int8_t const gap) const {
        std::stringstream obj_str;
        if (this->is_matrix()) {
            print_helper(obj_str, *this, 0);
            return obj_str.str();
        }

        auto const space_string = generate_spaces(gap);
        obj_str << space_string << "[\n";
        print_helper(obj_str, *this, static_cast<int8_t>(gap + 4));
        obj_str << space_string << "]\n";

        return obj_str.str();
    }
}
