//
// Created by sriram on 2/8/25.
//

#include <random>
#include <gtest/gtest.h>
#include "tensor.h"
#include "enums.h"

// TODO Finish testing tensor make GEMV have less redirects

TEST(TensorTestFunc, test_default_constructor) {
    cobraml::core::Tensor const tens;
    ASSERT_EQ(tens.get_dtype(), cobraml::core::INVALID);
    ASSERT_EQ(tens.get_device(), cobraml::core::CPU);
    ASSERT_TRUE(tens.get_shape().empty());
}

TEST(TensorTestFunc, test_constructor) {
    cobraml::core::Tensor const tens({3, 10, 20}, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_EQ(tens.get_dtype(), cobraml::core::CPU);
    ASSERT_EQ(tens.get_device(), cobraml::core::INT8);
    ASSERT_EQ(tens.len(), 600);

    ASSERT_THROW(
        cobraml::core::Tensor({0, 1}, cobraml::core::CPU, cobraml::core::INT8),
        std::runtime_error);

    cobraml::core::Tensor const tens2({10}, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_EQ(tens2.get_dtype(), cobraml::core::CPU);
    ASSERT_EQ(tens2.get_device(), cobraml::core::INT8);
    ASSERT_EQ(tens2.len(), 10);
    ASSERT_EQ(tens2.get_shape()[0], 1);
    ASSERT_EQ(tens2.get_shape()[1], 10);
}

#define IS_EQ(p1, p2, len){\
    for (size_t i = 0; i < len; ++i){\
        ASSERT_EQ(p1[i], p2[i]);\
    }\
}\

TEST(TensorTestFunc, test_to_mat) {
    cobraml::core::Tensor const tens({10, 20}, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_TRUE(tens.is_matrix());
    auto const mat{tens.to_matrix()};
    auto [rows, columns] = mat.get_shape();
    ASSERT_EQ(10, rows);
    ASSERT_EQ(20, columns);

    mat[5][10].set_item<int8_t>(10);

    const auto *mat_buffer = cobraml::core::get_buffer<int8_t>(mat);
    const auto *tensor_buffer = cobraml::core::get_buffer<int8_t>(tens);

    IS_EQ(mat_buffer, tensor_buffer, 200);

    cobraml::core::Tensor const tens_vec({10}, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_TRUE(tens_vec.is_matrix());
    auto const mat2{tens_vec.to_matrix()};
}

TEST(TensorTestFunc, test_from_vector) {
    std::vector const f{1.5f, 2.2f, 3.18f, 2.789f, 1.378f, -2.567f, 9.000f, 1.f};
    cobraml::core::Tensor const tens{from_vector(f, {2, 4, 1}, cobraml::core::CPU)};

    ASSERT_FALSE(tens.is_matrix());
    auto &shape = tens.get_shape();
    ASSERT_EQ(2, shape[0]);
    ASSERT_EQ(4, shape[1]);
    ASSERT_EQ(1, shape[2]);
    ASSERT_EQ(tens.get_dtype(), cobraml::core::FLOAT32);
    ASSERT_EQ(tens.get_device(), cobraml::core::CPU);

    const auto *tensor_buffer = cobraml::core::get_buffer<float>(tens);
    IS_EQ(tensor_buffer, f.data(), 8);

    ASSERT_THROW(cobraml::core::Tensor const tens2{from_vector(f, {2, 1}, cobraml::core::CPU)}, std::runtime_error);
}

TEST(TensorTestFunc, index_operator) {
    std::vector const f{1.5f, 2.2f, 3.18f, 2.789f, 1.378f, -2.567f, 9.000f, 1.f};
    cobraml::core::Tensor const tens{from_vector(f, {2, 2, 2}, cobraml::core::CPU)};
    ASSERT_FALSE(tens.is_matrix());

    for (size_t i = 0; i < 2; ++i) {
        cobraml::core::Tensor const m = tens[i];
        ASSERT_TRUE(m.is_matrix());
        ASSERT_EQ(m.len(), 4);
        for (size_t j = 0; j < 2; ++j) {
            cobraml::core::Tensor const v = m[j];
            ASSERT_TRUE(v.is_matrix());
            ASSERT_EQ(v.len(), 2);
            for (size_t k{0}; k < 2; ++k) {
                cobraml::core::Tensor const s = v[k];
                ASSERT_EQ(s.len(), 1);
                ASSERT_TRUE(s.is_matrix());
                auto scal{s.item<float>()};
                ASSERT_EQ(scal, f[i * 4 + j * 2 + k]);
            }
        }
    }

    cobraml::core::Tensor const t1 = tens[0];
    ASSERT_EQ(t1.len(), 4);

    auto const mat{t1.to_matrix()};
    auto [rows, columns] = mat.get_shape();
    ASSERT_EQ(2, rows);
    ASSERT_EQ(2, columns);
}
