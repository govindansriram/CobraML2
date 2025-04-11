//
// Created by sriram on 3/24/25.
//

#include <random>
#include <gtest/gtest.h>
#include "brarray.h"
#include <cmath>


#define CHECK_GEMV(mat, vec, start, result, alpha, beta, sum, epsilon){\
    for (size_t i = 0; i < start.size(); ++i){\
        start[i] *= beta;\
        sum = 0;\
        for (size_t j = 0; j < vec.size(); ++j) {\
            sum += mat[i * vec.size() + j] * vec[j];\
        }\
        sum *= alpha;\
        sum += start[i];\
        if (std::abs(result[i] - sum) > epsilon) {\
            std::cout << "here------------------ " << i << std::endl;\
            std::cout << "expected: " << sum << std::endl;\
            std::cout << "row: " << i << std::endl;\
            std::cout << "received: " << result[i] << std::endl;\
            std::cout << ">: " << (result[i] > sum) << std::endl;\
            std::cout << (result[i] - sum) << std::endl;\
        }\
        ASSERT_TRUE(std::abs(result[i] - sum) < epsilon);\
    }\
}

constexpr float fl_choice[10]{
    1.113f, -1.27948f, 2.12323f, 1.f, -2.108f, -1.3452f, 1.91782f, -1.23232f, -1.58f, .00001f
};

constexpr double dl_choice[10]{
    1.113, -1.27948, 10000.12323, 1, -7.108, -1.3452, 1597.91782, -12.23232, -2.58, .00001
};

constexpr int int_choice[10]{
    1, -1, 10, -15, 7, -34, 97, -32, 58, 0
};

constexpr double udl_choice[10]{
    1.113, 1.279, 1., 1, 7.108, 1.3452, 15., 12.23232, 2.58, .1
};

constexpr float ufl_choice[10]{
    7.5f, 1.f, 2.12f, 13.f, 2.108f, 5.4f, 1.91f, 1.23f, 1.58f, .1f
};

static int one_to_10() {
    static std::mt19937 generator(std::random_device{}());
    static std::uniform_int_distribution distribution(0, 9);
    return distribution(generator);
}

#define FILL_VECTOR(arr, vector) for(size_t i = 0; i < vector.size(); ++i) vector[i] = arr[one_to_10()];

TEST(MathTestFunc, gemv_float64_kernel) {
    constexpr double choice[10]{
        1.113, -1.27948, 10000.12323, 1, -7.108, -1.3452, 1597.91782, -12.23232, -2.58, .00001
    };

    std::vector vec1(7, 0.0);
    FILL_VECTOR(choice, vec1);

    std::vector vec2(3987, 0.0);
    FILL_VECTOR(choice, vec2);

    std::vector mat1(8 * 7, 0.0);
    FILL_VECTOR(choice, mat1);

    std::vector mat2(1234 * 3987, 0.0);
    FILL_VECTOR(choice, mat2);

    constexpr double alpha = 2.234;
    constexpr double beta = 0.0023;

    const std::vector res1(8, -0.2341);
    auto res_1_copy = res1;

    const std::vector res2(1234, 2.892);
    auto res_2_copy = res2;

    cobraml::core::Brarray const c_vec1(cobraml::core::CPU, cobraml::core::FLOAT64, {7}, vec1);
    cobraml::core::Brarray const c_vec2(cobraml::core::CPU, cobraml::core::FLOAT64, {3987}, vec2);
    cobraml::core::Brarray const c_mat1(cobraml::core::CPU, cobraml::core::FLOAT64, {8, 7}, mat1);
    cobraml::core::Brarray const c_mat2(cobraml::core::CPU, cobraml::core::FLOAT64, {1234, 3987}, mat2);
    cobraml::core::Brarray c_res1(cobraml::core::CPU, cobraml::core::FLOAT64, {8}, res1);
    cobraml::core::Brarray c_res2(cobraml::core::CPU, cobraml::core::FLOAT64, {1234}, res2);

    gemv(c_res1, c_mat1, c_vec1, alpha, beta);
    const auto *res1_buff = c_res1.get_buffer<double>();

    gemv(c_res2, c_mat2, c_vec2, alpha, beta);
    const auto *res2_buff = c_res2.get_buffer<double>();

    double sum;

    constexpr double EPSILON = 1e-3;
    CHECK_GEMV(mat1, vec1, res_1_copy, res1_buff, alpha, beta, sum, EPSILON);
    CHECK_GEMV(mat2, vec2, res_2_copy, res2_buff, alpha, beta, sum, EPSILON);


    // test single element multiplication of any single element

    auto s_vec1{c_vec1[0]};
    auto s_mat1{cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT64, {1, 1}, std::vector<double>{-12.2})};
    auto s_res1{cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT64, {1})};

    gemv(s_res1, s_mat1, s_vec1, 1.0, 1.0);

    ASSERT_TRUE(std::abs(s_res1.item<double>() - (-12.2 * s_vec1.item<double>())) < EPSILON);
}

TEST(MathTestFunc, gemv_float32_kernel) {
    constexpr float choice[10]{
        1.113f, -1.27948f, 2.12323f, 1.f, -2.108f, -1.3452f, 1.91782f, -1.23232f, -1.58f, .00001f
    };

    std::vector vec1(7, 0.0f);
    FILL_VECTOR(choice, vec1);

    std::vector vec2(3987, 0.0f);
    FILL_VECTOR(choice, vec2);

    std::vector mat1(8 * 7, 0.0f);
    FILL_VECTOR(choice, mat1);

    std::vector mat2(1231 * 3987, 0.0f);
    FILL_VECTOR(choice, mat2);

    constexpr float alpha = 2.234f;
    constexpr float beta = 0.0023f;

    const std::vector res1(8, -0.2341f);
    auto res_1_copy = res1;

    const std::vector res2(1231, 2.892f);
    auto res_2_copy = res2;

    cobraml::core::Brarray const c_vec1(cobraml::core::CPU, cobraml::core::FLOAT32, {7}, vec1);
    cobraml::core::Brarray const c_vec2(cobraml::core::CPU, cobraml::core::FLOAT32, {3987}, vec2);
    cobraml::core::Brarray const c_mat1(cobraml::core::CPU, cobraml::core::FLOAT32, {8, 7}, mat1);
    cobraml::core::Brarray const c_mat2(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat2);
    cobraml::core::Brarray c_res1(cobraml::core::CPU, cobraml::core::FLOAT32, {8}, res1);
    cobraml::core::Brarray c_res2(cobraml::core::CPU, cobraml::core::FLOAT32, {1231}, res2);

    gemv(c_res1, c_mat1, c_vec1, alpha, beta);
    const auto *res1_buff = c_res1.get_buffer<float>();

    gemv(c_res2, c_mat2, c_vec2, alpha, beta);
    const auto *res2_buff = c_res2.get_buffer<float>();

    float sum;

    constexpr float EPSILON = 1e-1f;
    CHECK_GEMV(mat1, vec1, res_1_copy, res1_buff, alpha, beta, sum, EPSILON);
    CHECK_GEMV(mat2, vec2, res_2_copy, res2_buff, alpha, beta, sum, EPSILON);

    // test single element multiplication of any single element

    auto s_vec1{c_vec1[0]};
    auto s_mat1{cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 1}, std::vector<float>{-12.2})};
    auto s_res1{cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {1})};

    gemv(s_res1, s_mat1, s_vec1, 1.0f, 1.0f);

    ASSERT_TRUE(std::abs(s_res1.item<float>() - (-12.2 * s_vec1.item<float>())) < EPSILON);
}

#define ELEMENT_WISE_OPERATION_S(tensor_vector, scalar, result_ptr, row_stride, total_rows, row_length, operation, dtype){\
    for (size_t i = 0; i < total_rows; ++i){\
        for (size_t j = 0; j < row_length; ++j){\
            dtype expected{operation(tensor_vector[i * row_length + j], scalar)};\
            ASSERT_EQ(expected, result_ptr[row_stride * i + j]);\
        }\
    }\
}

#define ELEMENT_WISE_OPERATION_MM(matrix_vector_1, matrix_vector_2, result_ptr, row_stride, total_rows, row_length, operation, dtype){\
    for (size_t i = 0; i < total_rows; ++i){\
        for (size_t j = 0; j < row_length; ++j){\
            dtype expected{operation(matrix_vector_1[i * row_length + j], matrix_vector_2[i * row_length + j])};\
            ASSERT_EQ(expected, result_ptr[row_stride * i + j]);\
        }\
    }\
}

#define ELEMENT_WISE_OPERATION_TV(tensor_vector, vector_vector, result_ptr, row_stride, total_rows, row_length, operation, dtype){\
    for (size_t i = 0; i < total_rows; ++i){\
        for (size_t j = 0; j < row_length; ++j){\
            dtype expected{operation(tensor_vector[i * row_length + j], vector_vector[j])};\
            ASSERT_EQ(expected, result_ptr[row_stride * i + j]);\
        }\
    }\
}

#define MULT(x, y) ((x) * (y))

TEST(MathTestFunc, mult) {

    const std::vector<float> tensor{
        1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36
    };

    const std::vector vector{10.0f, 11.0f, 12.0f};
    const std::vector scalar{21.23f};

    const std::vector expanded_vector_1{
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,

        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,

        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f
    };

    cobraml::core::Brarray b_tensor(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 1, 36}, tensor);
    cobraml::core::Brarray b_vec(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 1, 1, 1}, vector);

    cobraml::core::Brarray result{b_tensor * 21.23f};
    std::vector<size_t> expected_size{1, 1, 36};
    ASSERT_EQ(result.get_shape(), expected_size);
    ELEMENT_WISE_OPERATION_S(tensor, 21.23f, result.get_buffer<float>(), 40, 1, 36, MULT, float);

    result = b_tensor * 21;
    ASSERT_EQ(result.get_shape(), expected_size);
    ELEMENT_WISE_OPERATION_S(tensor, 21.f, result.get_buffer<float>(), 40, 1, 36, MULT, float);

    result = b_tensor * b_vec;
    expected_size = {3, 1, 1, 36};
    ASSERT_EQ(result.get_shape(), expected_size);
    ELEMENT_WISE_OPERATION_MM(
        tensor,
        std::vector(expanded_vector_1.begin(), expanded_vector_1.begin() + 36),
        result[0][0][0].get_buffer<float>(), 40, 1, 36, MULT, float);

    ELEMENT_WISE_OPERATION_MM(
        tensor,
        std::vector(expanded_vector_1.begin() + 36, expanded_vector_1.begin() + 72),
        result[1][0][0].get_buffer<float>(), 40, 1, 36, MULT, float);

    ELEMENT_WISE_OPERATION_MM(
        tensor,
        std::vector(expanded_vector_1.begin() + 72, expanded_vector_1.end()),
        result[2][0][0].get_buffer<float>(), 40, 1, 36, MULT, float);

    b_tensor = cobraml::core::Brarray(
        cobraml::core::CPU,
        cobraml::core::FLOAT32,
        {1, 3, 4, 3},
        tensor);

    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {3}, vector);
    result = b_vec * b_tensor;
    expected_size = {1, 3, 4, 3};
    ASSERT_EQ(result.get_shape(), expected_size);

    ELEMENT_WISE_OPERATION_TV(
        tensor,
        vector,
        result.get_buffer<float>(), 8, 12, 3, MULT, float);

    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 3, 1, 1}, vector);

    const std::vector expanded_vector_2{
        10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,

        11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,

        12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,
    };

    result = b_vec * b_tensor;
    expected_size = {1, 3, 4, 3};
    ASSERT_EQ(result.get_shape(), expected_size);

    ELEMENT_WISE_OPERATION_MM(
        tensor,
        std::vector(expanded_vector_2.begin(), expanded_vector_2.begin() + 12),
        result[0][0].get_buffer<float>(), 8, 4, 3, MULT, float);

    ELEMENT_WISE_OPERATION_MM(
        std::vector<float>(tensor.begin() + 12, tensor.begin() + 24),
        std::vector(expanded_vector_2.begin() + 12, expanded_vector_2.begin() + 24),
        result[0][1].get_buffer<float>(), 8, 4, 3, MULT, float);

    ELEMENT_WISE_OPERATION_MM(
        std::vector<float>(tensor.begin() + 24, tensor.end()),
        std::vector(expanded_vector_2.begin() + 24, expanded_vector_2.end()),
        result[0][2].get_buffer<float>(), 8, 4, 3, MULT, float);

    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 4, 3}, expanded_vector_2);
    b_tensor = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 4, 3}, tensor);

    result = b_vec * b_tensor;
    expected_size = {3, 4, 3};
    ASSERT_EQ(result.get_shape(), expected_size);
    ELEMENT_WISE_OPERATION_MM(
        tensor,
        expanded_vector_2,
        result.get_buffer<float>(), 8, 12, 3, MULT, float);

    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, {6});
    ASSERT_THROW(b_vec * b_tensor, std::runtime_error);
    ASSERT_THROW(b_vec * 10.10, std::runtime_error);
}

TEST(MathTestFunc, imult) {

    const std::vector<float> tensor{
        1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36
    };

    const std::vector vector{10.0f, 11.0f, 12.0f};
    const std::vector scalar{21.23f};

    const std::vector expanded_vector_1{
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,
        10.0f,  10.0f,  10.0f,  10.0f,  10.0f,  10.0f,

        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,
        11.0f,  11.0f,  11.0f,  11.0f,  11.0f,  11.0f,

        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
        12.0f,  12.0f,  12.0f,  12.0f,  12.0f,  12.0f,
    };

    cobraml::core::Brarray b_tensor(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 1, 36}, tensor);
    cobraml::core::Brarray b_vec(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 1, 1}, vector);

    imult(b_tensor, 21.23f);

    std::vector<size_t> expected_size{1, 1, 36};
    ASSERT_EQ(b_tensor.get_shape(), expected_size);
    ELEMENT_WISE_OPERATION_S(tensor, 21.23f, b_tensor.get_buffer<float>(), 40, 1, 36, MULT, float);

    b_tensor = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 1, 36}, tensor);
    imult(b_tensor, 10);
    ELEMENT_WISE_OPERATION_S(tensor, 10.f, b_tensor.get_buffer<float>(), 40, 1, 36, MULT, float);


    b_tensor = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 2, 6}, tensor);

    imult(b_tensor, b_vec);

    expected_size = {3, 2, 6};
    ASSERT_EQ(b_tensor.get_shape(), expected_size);
    ELEMENT_WISE_OPERATION_MM(
        std::vector<float>(tensor.begin(), tensor.begin() + 12),
        std::vector(expanded_vector_1.begin(), expanded_vector_1.begin() + 12),
        b_tensor[0].get_buffer<float>(), 8, 2, 6, MULT, float);

    ELEMENT_WISE_OPERATION_MM(
        std::vector<float>(tensor.begin() + 12, tensor.begin() + 24),
        std::vector(expanded_vector_1.begin() + 12, expanded_vector_1.begin() + 24),
        b_tensor[1].get_buffer<float>(), 8, 2, 6, MULT, float);

    ELEMENT_WISE_OPERATION_MM(
        std::vector<float>(tensor.begin() + 24, tensor.end()),
        std::vector(expanded_vector_1.begin() + 24, expanded_vector_1.end()),
        b_tensor[2].get_buffer<float>(), 8, 2, 6, MULT, float);

    b_tensor = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {6, 6}, tensor);
    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {6, 6}, expanded_vector_1);

    imult(b_tensor, b_vec);

    ELEMENT_WISE_OPERATION_MM(
        tensor,
        expanded_vector_1,
        b_tensor.get_buffer<float>(), 8, 6, 6, MULT, float);

    b_tensor = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 6, 2, 1}, tensor);
    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {6, 6}, expanded_vector_1);
    ASSERT_THROW(imult(b_tensor, b_vec), std::runtime_error);
    b_vec = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 36}, expanded_vector_1);
    ASSERT_THROW(imult(b_tensor, b_vec), std::runtime_error);
}
