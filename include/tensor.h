//
// Created by sriram on 2/8/25.
//

#ifndef TENSOR_H
#define TENSOR_H
#include "barray.h"
#include "matrix.h"

namespace cobraml::core {

    class Tensor final : public Barray{
        std::vector<size_t> shape;
        std::string generate_description() const override;

    public:

        /**
         * constructor that creates a zerod tensor of shape (rows, columns)
         * @param shape the shape of the tensor
         * @param device the device of the tensor being constructed
         * @param dtype the dtype of the tensor being constructed
         */
        Tensor(std::vector<size_t> shape, Device device, Dtype dtype);

        Tensor();
        Tensor(Tensor const &other);
        Tensor& operator=(const Tensor& other);
        Tensor operator[] (size_t index) const;
        std::string to_string(int8_t gap) const override;

        /**
         * @return True if tensor qualifies as a matrix
         */
        [[nodiscard]] bool is_matrix() const;

        /**
        * @return the shape of the tensor
        */
        [[nodiscard]] const std::vector<size_t>& get_shape() const;

        // /**
        //  * prints the contents of the matrix in tabular format
        //  * @param show_description
        //  */
        // void print(bool show_description) const override;

        [[nodiscard]] Matrix to_matrix() const;

        ~Tensor() override;

        // Start of the Friend API
        template<typename T>
        friend Tensor from_vector(const std::vector<T> &vec, std::vector<size_t> shape, Device device);
    };

    template<typename T>
    Tensor from_vector(const std::vector<T> &vec, std::vector<size_t> shape, Device const device) {
        constexpr Dtype dtype{get_dtype_from_type<T>::type};
        is_invalid(dtype);
        size_t prod{1};

        for (size_t const &i: shape) {
            prod *= i;
        }

        if (prod != vec.size()) {
            throw std::runtime_error("shape and vector length do not match");
        }

        Tensor ret(std::move(shape), device, dtype);

        ret.copy_vector(vec);

        return ret;
    }
}

#endif //TENSOR_H
