//
// Created by sriram on 3/24/25.
//

#include <random>
#include <gtest/gtest.h>
#include "brarray.h"

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

static int one_to_10() {
    static std::mt19937 generator(std::random_device{}());
    static std::uniform_int_distribution distribution(0, 9);
    return distribution(generator);
}

#define FILL_VECTOR(arr, vector) for(size_t i = 0; i < vector.size(); ++i) vector[i] = arr[one_to_10()];

TEST(MatrixTestFunc, gemv_float64_kernel) {
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
}

TEST(MatrixTestFunc, gemv_float32_kernel) {
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

    constexpr float EPSILON = 1e-2f;
    CHECK_GEMV(mat1, vec1, res_1_copy, res1_buff, alpha, beta, sum, EPSILON);
    CHECK_GEMV(mat2, vec2, res_2_copy, res2_buff, alpha, beta, sum, EPSILON);
}
