//
// Created by sriram on 4/21/25.
//
#include <gtest/gtest.h>
#include "brarray.h"

TEST(CudaArrayTestFunctionals, test_device) {
    const std::vector<float> vec{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    const cobraml::core::Brarray tensor(
        cobraml::core::CUDA,
        cobraml::core::FLOAT32,
        {2, 4, 3}, vec);

    for (size_t i{0}; i < tensor.get_shape()[0]; ++i) {
        auto matrix{tensor[i]};
        for (size_t j{0}; j < matrix.get_shape()[0]; ++j) {
            auto vector{matrix[j]};
            for (size_t k{0}; k < vector.get_shape()[0]; ++k) {
                auto scalar{vector[k]};
                float expected{vec[i * 12 + j * 3 + k]};
                ASSERT_EQ(scalar.item<float>(), expected);
            }
        }
    }

    auto scal{tensor[0][2][2]};
    scal.set_item(10.3f);

    auto res = cobraml::core::gemv(tensor[0], tensor[0][0], 1.0f, 1.0f);


    std::cout << res;
    std::cout << tensor;
}

TEST(CudaArrayTestFunctionals, eq) {
    std::vector vec(100000, 0);

    int count{0};
    for (auto &item: vec) {
        item = count;
        count += 1;
    }

    const cobraml::core::Brarray arr(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {10, 100, 100},
        vec);

    ASSERT_FALSE(arr[1] == arr[0]);
    ASSERT_TRUE(arr[0] == arr[0]);
    ASSERT_TRUE(arr == arr);

    vec[99999] = -1;

    const cobraml::core::Brarray arr2(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {10, 100, 100},
        vec);

    ASSERT_FALSE(arr == arr2);
    ASSERT_TRUE(arr[0] == arr2[0]);
}

TEST(CudaArrayTestFunctionals, gemm) {
    const std::vector<int> vec{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27
    };

    const std::vector<int> eye{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    const std::vector<int> mat{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };

    const cobraml::core::Brarray tensor(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {3, 3, 3}, vec);

    const cobraml::core::Brarray ident(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {3, 3}, eye);

    const cobraml::core::Brarray matr(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {3, 4}, mat);

    auto res{cobraml::core::gemm(tensor[0], ident, 1, 1)};
    ASSERT_TRUE(res == tensor[0]);

    res = cobraml::core::gemm(tensor[1], matr, 1, 1);
    const std::vector<size_t> exp{3, 4};
    ASSERT_EQ(res.get_shape(), exp);
    ASSERT_EQ(res.get_dtype(), cobraml::core::INT32);
    ASSERT_EQ(res.get_device(), cobraml::core::CUDA);

    // Associativity
    res = cobraml::core::gemm(tensor[1], tensor[2], 1, 1);
    res = cobraml::core::gemm(res, matr, 1, 1);

    auto res2{cobraml::core::gemm(tensor[2], matr, 1, 1)};
    res2 = cobraml::core::gemm(tensor[1], res2, 1, 1);

    ASSERT_EQ(res, res2);

    // test on matrix larger than 16 x 16

    std::vector<int> large(24 * 49, 0);
    std::vector<int> med(49 * 27, 0);
    std::vector<int> fin(24 * 27, 0);


    for (int i = 0; i < 24 * 49; ++i)
        large[i] = 1;

    for (int i = 0; i < 49 * 27; ++i)
        med[i] = i % 27;

    for (int i = 0; i < 24 * 27; ++i)
        fin[i] = (i % 27) * 49;

    const cobraml::core::Brarray large_ten(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {24, 49}, large);

    const cobraml::core::Brarray med_ten(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {49, 27}, med);

    const cobraml::core::Brarray final_ten(
        cobraml::core::CUDA,
        cobraml::core::INT32,
        {24, 27}, fin);

    res = cobraml::core::gemm(large_ten, med_ten, 1, 1);

    ASSERT_TRUE(res == final_ten);
}
