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

#define ELEMENT_WISE_OPERATION_SCALAR(tensor_vector, scalar, result_ptr, row_stride, total_rows, row_length, operation, dtype){\
    for (size_t i = 0; i < total_rows; ++i){\
        for (size_t j = 0; j < row_length; ++j){\
            dtype expected{operation(tensor_vector[i * row_length + j], scalar)};\
            ASSERT_EQ(expected, result_ptr[row_stride * i + j]);\
        }\
    }\
}

#define MULT(x, y) ((x) * (y))

TEST(MathTestFunc, hadamard_product) {

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

    cobraml::core::Brarray b_tensor(cobraml::core::CPU, cobraml::core::FLOAT32, {1, 1, 36}, tensor);
    cobraml::core::Brarray b_vec(cobraml::core::CPU, cobraml::core::FLOAT32, {3, 1, 1, 1}, vector);
    cobraml::core::Brarray b_scal(cobraml::core::CPU, cobraml::core::FLOAT32, {1}, scalar);

    const cobraml::core::Brarray result{b_tensor * b_scal};
    // std::cout << result;
    ELEMENT_WISE_OPERATION_SCALAR(tensor, 21.23f, result.get_buffer<float>(), 40, 1, 36, MULT, float);

    // const std::vector<size_t> expected_shape{5, 10, 3};
    // ASSERT_EQ(result.get_shape(), expected_shape);
}

#define CHECK_POW(type, b_1, b_2, result, row_stride, columns){\
    for (size_t i = 0; i < b_1.size(); ++i){\
        size_t row{i / columns};\
        size_t column{i % columns};\
        ASSERT_EQ(static_cast<type>(pow(b_1[i], b_2[i])), result[row_stride * row + column]);\
    }\
}
//
// TEST(MathTestFunc, element_wise_power) {
//     std::vector vec1(50, 0);
//     std::vector vec2(50, 0);
//     FILL_VECTOR(int_choice, vec1);
//     FILL_VECTOR(int_choice, vec2);
//
//     std::vector mat1(1231 * 3987, 0.0f);
//     std::vector mat2(1231 * 3987, 0.0f);
//     FILL_VECTOR(ufl_choice, mat2);
//     FILL_VECTOR(ufl_choice, mat1);
//
//     std::vector tensor1(4 * 8 * 99, 0.0);
//     std::vector tensor2(4 * 8 * 99, 0.0);
//     FILL_VECTOR(udl_choice, tensor1);
//     FILL_VECTOR(udl_choice, tensor2);
//
//
//     cobraml::core::Brarray const b_vec1(cobraml::core::CPU, cobraml::core::INT32, {50}, vec1);
//     cobraml::core::Brarray const b_vec2(cobraml::core::CPU, cobraml::core::INT32, {50}, vec2);
//
//     cobraml::core::Brarray const b_mat1(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat1);
//     cobraml::core::Brarray const b_mat2(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat2);
//
//     cobraml::core::Brarray const b_ten1(cobraml::core::CPU, cobraml::core::FLOAT64, {4, 8, 99}, tensor1);
//     cobraml::core::Brarray const b_ten2(cobraml::core::CPU, cobraml::core::FLOAT64, {4, 8, 99}, tensor2);
//
//     cobraml::core::Brarray const b_vec3{cobraml::core::pow(b_vec1, b_vec2)};
//     const int * vec_p{b_vec3.get_buffer<int>()};
//     CHECK_POW(int, vec1, vec2, vec_p, 0, 50);
//
//     cobraml::core::Brarray const b_mat3{cobraml::core::pow(b_mat1, b_mat2)};
//     const float * mat_p{b_mat3.get_buffer<float>()};
//     CHECK_POW(float, mat1, mat2, mat_p, 3992, 3987);
//
//     cobraml::core::Brarray const b_ten3{cobraml::core::pow(b_ten1, b_ten2)};
//     const double * ten_p{b_ten3.get_buffer<double>()};
//     CHECK_POW(double, tensor1, tensor2, ten_p, 100, 99);
//
//     ASSERT_THROW(cobraml::core::pow(b_ten2, b_vec1), std::runtime_error);
//
//     std::vector<int8_t> tensor8(4 * 8 * 99, 0);
//     cobraml::core::Brarray const b_ten8(cobraml::core::CPU, cobraml::core::INT8, {4, 8, 99}, tensor8);
//     ASSERT_THROW(cobraml::core::pow(b_ten2, b_ten8), std::runtime_error);
//
//     cobraml::core::Brarray empty;
//     ASSERT_THROW(cobraml::core::pow(empty, empty), std::runtime_error);
//
//     cobraml::core::Brarray const diff_device(cobraml::core::GPU, cobraml::core::FLOAT32, {4, 8, 99});
//     ASSERT_THROW(cobraml::core::pow(b_ten2, diff_device), std::runtime_error);
// }
//
#define CHECK_ADD(b_1, b_2, result, row_stride, columns){\
    for (size_t i = 0; i < b_1.size(); ++i){\
        size_t row{i / columns};\
        size_t column{i % columns};\
        ASSERT_EQ(b_1[i] + b_2[i], result[row_stride * row + column]);\
    }\
}
//
// TEST(MathTestFunc, element_wise_add) {
//     std::vector vec1(50, 0.0f);
//     std::vector vec2(50, 0.0f);
//     FILL_VECTOR(fl_choice, vec1);
//     FILL_VECTOR(fl_choice, vec2);
//
//     std::vector mat1(1231 * 3987, 0.0f);
//     std::vector mat2(1231 * 3987, 0.0f);
//     FILL_VECTOR(fl_choice, mat2);
//     FILL_VECTOR(fl_choice, mat1);
//
//     std::vector tensor1(4 * 8 * 99, 0.0f);
//     std::vector tensor2(4 * 8 * 99, 0.0f);
//     FILL_VECTOR(fl_choice, tensor1);
//     FILL_VECTOR(fl_choice, tensor2);
//
//     cobraml::core::Brarray const b_vec1(cobraml::core::CPU, cobraml::core::FLOAT32, {50}, vec1);
//     cobraml::core::Brarray const b_vec2(cobraml::core::CPU, cobraml::core::FLOAT32, {50}, vec2);
//     cobraml::core::Brarray const b_mat1(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat1);
//     cobraml::core::Brarray const b_mat2(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat2);
//     cobraml::core::Brarray const b_ten1(cobraml::core::CPU, cobraml::core::FLOAT32, {4, 8, 99}, tensor1);
//     cobraml::core::Brarray const b_ten2(cobraml::core::CPU, cobraml::core::FLOAT32, {4, 8, 99}, tensor2);
//
//     cobraml::core::Brarray const b_vec3{b_vec1 + b_vec2};
//     cobraml::core::Brarray const b_vec31{cobraml::core::add(b_vec1, b_vec2)};
//
//     const float * vec_p{b_vec3.get_buffer<float>()};
//     const float * vec_p1{b_vec31.get_buffer<float>()};
//
//     CHECK_ADD(vec1, vec2, vec_p, 0, 50);
//     CHECK_ADD(vec1, vec2, vec_p1, 0, 50);
//
//     cobraml::core::Brarray const b_mat3{b_mat1 + b_mat2};
//     cobraml::core::Brarray const b_mat31{cobraml::core::add(b_mat1, b_mat2)};
//
//     const float * mat_p{b_mat3.get_buffer<float>()};
//     const float * mat_p1{b_mat31.get_buffer<float>()};
//
//     CHECK_ADD(mat1, mat2, mat_p, 3992, 3987);
//     CHECK_ADD(mat1, mat2, mat_p1, 3992, 3987);
//
//     cobraml::core::Brarray const b_ten3{b_ten1 + b_ten2};
//     cobraml::core::Brarray const b_ten31{cobraml::core::add(b_ten1, b_ten2)};
//
//     const float * ten_p{b_ten3.get_buffer<float>()};
//     const float * ten_p1{b_ten31.get_buffer<float>()};
//
//     CHECK_ADD(tensor1, tensor2, ten_p, 104, 99);
//     CHECK_ADD(tensor1, tensor2, ten_p1, 104, 99);
//
//     ASSERT_THROW(b_ten2 + b_vec1, std::runtime_error);
//     ASSERT_THROW(cobraml::core::add(b_ten1, b_vec1), std::runtime_error);
//
//     std::vector<int8_t> tensor8(4 * 8 * 99, 0);
//     cobraml::core::Brarray const b_ten8(cobraml::core::CPU, cobraml::core::INT8, {4, 8, 99}, tensor8);
//     ASSERT_THROW(b_ten2 + b_ten8, std::runtime_error);
//     ASSERT_THROW(cobraml::core::add(b_ten1, b_ten8), std::runtime_error);
//
//     cobraml::core::Brarray empty;
//     ASSERT_THROW(empty + empty, std::runtime_error);
//     ASSERT_THROW(cobraml::core::add(empty, empty), std::runtime_error);
//
//     cobraml::core::Brarray const diff_device(cobraml::core::GPU, cobraml::core::FLOAT32, {4, 8, 99});
//     ASSERT_THROW(b_ten2 + diff_device, std::runtime_error);
//     ASSERT_THROW(cobraml::core::add(b_ten2, diff_device), std::runtime_error);
// }
//
#define CHECK_SUB(b_1, b_2, result, row_stride, columns){\
    for (size_t i = 0; i < b_1.size(); ++i){\
        size_t row{i / columns};\
        size_t column{i % columns};\
        ASSERT_EQ(b_1[i] - b_2[i], result[row_stride * row + column]);\
    }\
}

// TEST(MathTestFunc, element_wise_sub) {
//     std::vector vec1(50, 0.0f);
//     std::vector vec2(50, 0.0f);
//     FILL_VECTOR(fl_choice, vec1);
//     FILL_VECTOR(fl_choice, vec2);
//
//     std::vector mat1(1231 * 3987, 0.0f);
//     std::vector mat2(1231 * 3987, 0.0f);
//     FILL_VECTOR(fl_choice, mat2);
//     FILL_VECTOR(fl_choice, mat1);
//
//     std::vector tensor1(4 * 8 * 99, 0.0f);
//     std::vector tensor2(4 * 8 * 99, 0.0f);
//     FILL_VECTOR(fl_choice, tensor1);
//     FILL_VECTOR(fl_choice, tensor2);
//
//     cobraml::core::Brarray const b_vec1(cobraml::core::CPU, cobraml::core::FLOAT32, {50}, vec1);
//     cobraml::core::Brarray const b_vec2(cobraml::core::CPU, cobraml::core::FLOAT32, {50}, vec2);
//     cobraml::core::Brarray const b_mat1(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat1);
//     cobraml::core::Brarray const b_mat2(cobraml::core::CPU, cobraml::core::FLOAT32, {1231, 3987}, mat2);
//     cobraml::core::Brarray const b_ten1(cobraml::core::CPU, cobraml::core::FLOAT32, {4, 8, 99}, tensor1);
//     cobraml::core::Brarray const b_ten2(cobraml::core::CPU, cobraml::core::FLOAT32, {4, 8, 99}, tensor2);
//
//     cobraml::core::Brarray const b_vec3{b_vec1 - b_vec2};
//     cobraml::core::Brarray const b_vec31{cobraml::core::sub(b_vec1, b_vec2)};
//
//     const float * vec_p{b_vec3.get_buffer<float>()};
//     const float * vec_p1{b_vec31.get_buffer<float>()};
//
//     CHECK_SUB(vec1, vec2, vec_p, 0, 50);
//     CHECK_SUB(vec1, vec2, vec_p1, 0, 50);
//
//     cobraml::core::Brarray const b_mat3{b_mat1 - b_mat2};
//     cobraml::core::Brarray const b_mat31{cobraml::core::sub(b_mat1, b_mat2)};
//
//     const float * mat_p{b_mat3.get_buffer<float>()};
//     const float * mat_p1{b_mat31.get_buffer<float>()};
//
//     CHECK_SUB(mat1, mat2, mat_p, 3992, 3987);
//     CHECK_SUB(mat1, mat2, mat_p1, 3992, 3987);
//
//     cobraml::core::Brarray const b_ten3{b_ten1 - b_ten2};
//     cobraml::core::Brarray const b_ten31{cobraml::core::sub(b_ten1, b_ten2)};
//
//     const float * ten_p{b_ten3.get_buffer<float>()};
//     const float * ten_p1{b_ten31.get_buffer<float>()};
//
//     CHECK_SUB(tensor1, tensor2, ten_p, 104, 99);
//     CHECK_SUB(tensor1, tensor2, ten_p1, 104, 99);
//
//     ASSERT_THROW(b_ten2 - b_vec1, std::runtime_error);
//     ASSERT_THROW(cobraml::core::sub(b_ten1, b_vec1), std::runtime_error);
//
//     std::vector<int8_t> tensor8(4 * 8 * 99, 0);
//     cobraml::core::Brarray const b_ten8(cobraml::core::CPU, cobraml::core::INT8, {4, 8, 99}, tensor8);
//     ASSERT_THROW(b_ten2 - b_ten8, std::runtime_error);
//     ASSERT_THROW(cobraml::core::sub(b_ten1, b_ten8), std::runtime_error);
//
//     cobraml::core::Brarray empty;
//     ASSERT_THROW(empty - empty, std::runtime_error);
//     ASSERT_THROW(cobraml::core::sub(empty, empty), std::runtime_error);
//
//     cobraml::core::Brarray const diff_device(cobraml::core::GPU, cobraml::core::FLOAT32, {4, 8, 99});
//     ASSERT_THROW(b_ten2 - diff_device, std::runtime_error);
//     ASSERT_THROW(cobraml::core::sub(b_ten2, diff_device), std::runtime_error);
// }
