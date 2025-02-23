//
// Created by sriram on 12/15/24.
//


#include <random>
#include <gtest/gtest.h>
#include "matrix.h"
#include "enums.h"

#define CHECK_DP(mat, vec, start, result, alpha, beta, state, sum, epsilon){\
    for (size_t i = 0; i < mat.size(); ++i){\
        start[0][i] *= beta;\
        sum = 0;\
        for (size_t j = 0; j < vec[0].size(); ++j) {\
            sum += mat[i][j] * vec[0][j];\
        }\
        sum *= alpha;\
        sum += start[0][i];\
        if (std::abs(result[i] - sum) > epsilon) {\
            std::cout << "here------------------ " << i << std::endl;\
            std::cout << "expected: " << sum << std::endl;\
            std::cout << "received: " << result[i] << std::endl;\
            std::cout << ">: " << (result[i] > sum) << std::endl;\
            std::cout << (result[i] - sum) << std::endl;\
            state = false;\
            break;\
        }\
    }\
}

static int one_to_10() {
    static std::mt19937 generator(std::random_device{}());
    static std::uniform_int_distribution distribution(0, 9);
    return distribution(generator);
}

#define FILL(arr, matrix, rows, columns) {\
    for(size_t i = 0; i < rows; ++i){\
        for(size_t j = 0; j < columns; ++j){\
            matrix[i][j] = arr[one_to_10()];\
        }\
    }\
}

TEST(MatrixTestFunc, test_invalid_constructor) {
    ASSERT_THROW(
        cobraml::core::Matrix mat(10, 20, cobraml::core::CPU, cobraml::core::INVALID),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Matrix mat(0, 20, cobraml::core::CPU, cobraml::core::INT8),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Matrix mat(10, 0, cobraml::core::CPU, cobraml::core::INT8),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Matrix mat(0, 0, cobraml::core::CPU, cobraml::core::INT8),
        std::runtime_error);
}

TEST(MatrixTestFunc, test_valid_constructor) {
    cobraml::core::Matrix const mat(145, 112, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_EQ(mat.is_scalar(), false);
    ASSERT_EQ(mat.is_vector(), false);
    ASSERT_EQ(mat.get_shape().columns, 112);
    ASSERT_EQ(mat.get_shape().rows, 145);
    ASSERT_EQ(mat.get_device(), cobraml::core::CPU);
    ASSERT_EQ(mat.get_dtype(), cobraml::core::INT8);
}


TEST(MatrixTestFunc, test_is_vector) {
    cobraml::core::Matrix const vec(1, 10, cobraml::core::CPU, cobraml::core::INT8);
    cobraml::core::Matrix const mat(2, 10, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_EQ(vec.is_vector(), true);
    ASSERT_EQ(mat.is_vector(), false);
}


TEST(MatrixTestFunc, test_is_scalar) {
    cobraml::core::Matrix const scalar(1, 1, cobraml::core::CPU, cobraml::core::INT8);
    cobraml::core::Matrix const mat(2, 10, cobraml::core::CPU, cobraml::core::INT8);
    ASSERT_EQ(scalar.is_scalar(), true);
    ASSERT_EQ(mat.is_vector(), false);
}

TEST(MatrixTestFunc, test_default_constructor) {
    cobraml::core::Matrix const mat;
    ASSERT_EQ(mat.is_scalar(), false);
    ASSERT_EQ(mat.is_vector(), false);
    ASSERT_EQ(mat.get_shape().columns, 0);
    ASSERT_EQ(mat.get_shape().rows, 0);
    ASSERT_EQ(mat.get_device(), cobraml::core::CPU);
    ASSERT_THROW(cobraml::core::is_invalid(mat.get_dtype()), std::runtime_error);
}

TEST(MatrixTestFunc, test_copy_constuctor) {
    cobraml::core::Matrix const mat(5, 5, cobraml::core::CPU, cobraml::core::INT8);
    cobraml::core::Matrix const mat1{mat};

    const auto [rows, columns]{mat.get_shape()};
    const auto [rows1, columns1]{mat1.get_shape()};

    ASSERT_EQ(columns, columns1);
    ASSERT_EQ(rows, rows1);
    ASSERT_EQ(mat.get_dtype(), mat1.get_dtype());
    ASSERT_EQ(mat1.get_device(), mat.get_device());

    size_t const total{rows * columns};

    const int8_t *p{cobraml::core::get_buffer<int8_t>(mat)};
    const int8_t *p1{cobraml::core::get_buffer<int8_t>(mat1)};

    for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(p[0], p1[0]);
    }
}

TEST(MatrixTestFunc, test_copy_assignment_operator) {
    cobraml::core::Matrix const mat(5, 5, cobraml::core::CPU, cobraml::core::INT8);
    cobraml::core::Matrix mat1(10, 20, cobraml::core::CPU, cobraml::core::INT64);
    mat1 = mat;

    const auto [rows, columns]{mat.get_shape()};
    const auto [rows1, columns1]{mat1.get_shape()};

    ASSERT_EQ(columns, columns1);
    ASSERT_EQ(rows, rows1);
    ASSERT_EQ(mat.get_dtype(), mat1.get_dtype());
    ASSERT_EQ(mat1.get_device(), mat.get_device());

    size_t const total{rows * columns};

    const int8_t *p{cobraml::core::get_buffer<int8_t>(mat)};
    const int8_t *p1{cobraml::core::get_buffer<int8_t>(mat1)};

    for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(p[0], p1[0]);
    }
}

TEST(MatrixTestFunc, test_invalid_from_vector) {
    std::vector<std::vector<float> > const vec;
    ASSERT_THROW(const auto mat{from_vector(vec, cobraml::core::CPU)}, std::runtime_error);

    std::vector<std::vector<float> > const vec2{{}, {}};
    ASSERT_THROW(const auto mat{from_vector(vec2, cobraml::core::CPU)}, std::runtime_error);

    std::vector<std::vector<float> > const vec3{{1, 2, 3, 4}, {1, 2, 3}};
    ASSERT_THROW(const auto mat{from_vector(vec3, cobraml::core::CPU)}, std::runtime_error);
}

TEST(MatrixTestFunc, test_from_vector) {
    std::vector<std::vector<float> > const vec{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f}};
    const auto mat{from_vector(vec, cobraml::core::CPU)};
    const auto *buff{cobraml::core::get_buffer<float>(mat)};

    for (size_t i = 0; i < mat.get_shape().rows * mat.get_shape().columns; ++i) {
        ASSERT_EQ(static_cast<float>(i), buff[i]);
    }

    std::vector<std::vector<float> > const vec2{{0.0f, 1.0f, 2.0f}};
    const auto mat2{from_vector(vec2, cobraml::core::CPU)};
    const auto *buff2{cobraml::core::get_buffer<float>(mat2)};

    for (size_t i = 0; i < mat2.get_shape().rows * mat2.get_shape().columns; ++i) {
        ASSERT_EQ(static_cast<float>(i), buff2[i]);
    }
}

TEST(MatrixTestFunc, test_print) {
    std::vector<std::vector<float> > const vec{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f}};
    const auto mat{from_vector(vec, cobraml::core::CPU)};

    std::cout << mat;
    std::cout << "\n";

    const cobraml::core::Matrix mat2(40, 40, cobraml::core::CPU, cobraml::core::INT8);
    std::cout << mat2;
    std::cout << "\n";

    const cobraml::core::Matrix mat3(1, 10, cobraml::core::CPU, cobraml::core::INT8);
    std::cout << mat3;
    std::cout << "\n";

}


TEST(MatrixTestFunc, test_index_operator) {
    std::vector<std::vector<float> > const vec{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}, {6.0f, 7.0f, 8.0f}};
    const auto mat{from_vector(vec, cobraml::core::CPU)};

    auto [rows, columns]{mat.get_shape()};

    for (size_t i = 0; i < rows; ++i) {
        auto mv{mat[i]};
        ASSERT_TRUE(mv.is_vector());
        auto [mv_rows, mv_cols]{mat.get_shape()};
        ASSERT_EQ(mv_cols, 3);
        for (size_t j = 0; j < columns; ++j) {
            ASSERT_TRUE(mv[j].is_scalar());
            ASSERT_EQ(mv[j].item<float>(), vec[i][j]);
        }
    }

    ASSERT_THROW(mat[4], std::out_of_range);
    ASSERT_THROW(mat[0][10], std::out_of_range);

    auto scal{mat[2][1]};
    auto buff{cobraml::core::get_buffer<float>(scal)};
    ASSERT_EQ(buff[0], 7.0f);

    ASSERT_THROW(scal[1], std::out_of_range);
}

/**
 ************************************* TEST GEMV *************************************************
 */

TEST(MatrixTestFunc, test_invalid_gemv_vector) {
    cobraml::core::Matrix const mat(
        10, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    cobraml::core::Matrix vec(
        2, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    cobraml::core::Matrix res(
        1, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    constexpr float alpha = 2.1f;
    constexpr float beta = -1.1f;

    ASSERT_THROW(gemv(mat, vec, res, &alpha, &beta), std::runtime_error);

    const auto vec2 = cobraml::core::Matrix(1, 5, cobraml::core::CPU, cobraml::core::FLOAT32);

    ASSERT_THROW(gemv(mat, vec2, res, &alpha, &beta), std::runtime_error);
}

TEST(MatrixTestFunc, test_invalid_gemv_result) {
    cobraml::core::Matrix const mat(
        10, 20, cobraml::core::CPU, cobraml::core::FLOAT32);

    cobraml::core::Matrix const vec(
        1, 20, cobraml::core::CPU, cobraml::core::FLOAT32);

    cobraml::core::Matrix res(
        2, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    constexpr float alpha = 2.1f;
    constexpr float beta = -1.1f;

    ASSERT_THROW(gemv(mat, vec, res, &alpha, &beta), std::runtime_error);

    auto res2 = cobraml::core::Matrix(1, 5, cobraml::core::CPU, cobraml::core::FLOAT32);

    ASSERT_THROW(gemv(mat, vec, res2, &alpha, &beta), std::runtime_error);
}

TEST(MatrixTestFunc, test_invalid_gemv_dtype) {
    constexpr float alpha = 2.1f;
    constexpr float beta = -1.1f;

    cobraml::core::Matrix mat(
        10, 10, cobraml::core::CPU, cobraml::core::INT32);

    cobraml::core::Matrix vec(
        1, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    cobraml::core::Matrix res(
        1, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    ASSERT_THROW(gemv(mat, vec, res, &alpha, &beta), std::runtime_error);

    auto mat1 = cobraml::core::Matrix(10, 10, cobraml::core::CPU, cobraml::core::FLOAT32);
    auto vec1 = cobraml::core::Matrix(1, 10, cobraml::core::CPU, cobraml::core::INT32);

    ASSERT_THROW(gemv(mat1, vec1, res, &alpha, &beta), std::runtime_error);

    auto vec2 = cobraml::core::Matrix(10, 10, cobraml::core::CPU, cobraml::core::FLOAT32);
    auto res2 = cobraml::core::Matrix(1, 10, cobraml::core::CPU, cobraml::core::INT32);

    ASSERT_THROW(gemv(mat, vec2, res2, &alpha, &beta), std::runtime_error);
}

TEST(MatrixTestFunc, test_invalid_gemv_device) {
    constexpr float alpha = 2.1f;
    constexpr float beta = -1.1f;

    cobraml::core::Matrix mat(
        10, 10, cobraml::core::CPU_X, cobraml::core::FLOAT32);

    cobraml::core::Matrix vec(
        1, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    cobraml::core::Matrix res(
        1, 10, cobraml::core::CPU, cobraml::core::FLOAT32);

    ASSERT_THROW(gemv(mat, vec, res, &alpha, &beta), std::runtime_error);

    auto mat1 = cobraml::core::Matrix(10, 10, cobraml::core::CPU, cobraml::core::FLOAT32);
    auto vec1 = cobraml::core::Matrix(1, 10, cobraml::core::CPU_X, cobraml::core::FLOAT32);

    ASSERT_THROW(gemv(mat1, vec1, res, &alpha, &beta), std::runtime_error);

    auto vec2 = cobraml::core::Matrix(10, 10, cobraml::core::CPU, cobraml::core::FLOAT32);
    auto res2 = cobraml::core::Matrix(1, 10, cobraml::core::CPU_X, cobraml::core::FLOAT32);

    ASSERT_THROW(gemv(mat, vec2, res2, &alpha, &beta), std::runtime_error);
}


TEST(MatrixTestFunc, gemv_float64_kernel) {
    constexpr double choice[10]{
        1.113, -1.27948, 10000.12323, 1, -7.108, -1.3452, 1597.91782, -12.23232, -2.58, .00001
    };

    std::vector vec1(
        1, std::vector(5, 0.0)
    );

    FILL(choice, vec1, 1, 5);

    std::vector vec2(
        1, std::vector(3987, 0.0)
    );

    FILL(choice, vec2, 1, 3987);


    std::vector mat1(
        5, std::vector(5, 0.0)
    );

    FILL(choice, mat1, 5, 5);

    std::vector mat2(
        1234, std::vector(3987, 0.0)
    );

    FILL(choice, mat2, 1234, 3987);

    constexpr double alpha = 2.234;
    constexpr double beta = 0.0023;

    std::vector res1(
        1, std::vector(5, -0.2341)
    );

    auto res_1_copy = res1;

    std::vector res2(
        1, std::vector(1234, 2.892)
    );

    auto res_2_copy = res2;

    cobraml::core::Matrix const c_vec1 = cobraml::core::from_vector<double>(vec1, cobraml::core::CPU);
    cobraml::core::Matrix const c_vec2 = cobraml::core::from_vector<double>(vec2, cobraml::core::CPU);
    cobraml::core::Matrix const c_mat1 = cobraml::core::from_vector<double>(mat1, cobraml::core::CPU);
    cobraml::core::Matrix const c_mat2 = cobraml::core::from_vector<double>(mat2, cobraml::core::CPU);
    cobraml::core::Matrix c_res1 = cobraml::core::from_vector<double>(res1, cobraml::core::CPU);
    cobraml::core::Matrix c_res2 = cobraml::core::from_vector<double>(res2, cobraml::core::CPU);

    gemv(c_mat1, c_vec1, c_res1, alpha, beta);
    const auto *res1_buff = cobraml::core::get_buffer<double>(c_res1);

    bool state{true};
    double sum;

    constexpr double EPSILON = 1e-3;

    CHECK_DP(mat1, vec1, res_1_copy, res1_buff, alpha, beta, state, sum, EPSILON);
    ASSERT_EQ(state, true);

    gemv(c_mat2, c_vec2, c_res2, alpha, beta);
    const auto *res2_buff = cobraml::core::get_buffer<double>(c_res2);
    CHECK_DP(mat2, vec2, res_2_copy, res2_buff, alpha, beta, state, sum, EPSILON);
    ASSERT_EQ(state, true);
}
