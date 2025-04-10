//
// Created by sriram on 1/26/25.
//
#include <gtest/gtest.h>
#include "brarray.h"

TEST(ArrayTestFunctionals, test_dtype) {
    cobraml::core::Brarray arr(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {10});
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT8);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT16, {10});
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT16);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT32, {10});
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT32);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT64, {10});
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT64);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32, {10});
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::FLOAT32);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT64, {10});
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::FLOAT64);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INVALID, {10}),
        std::runtime_error);
}


TEST(ArrayTestFunctionals, test_device) {
    cobraml::core::Brarray arr(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {10});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU_X, cobraml::core::Dtype::INT8, {10});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU_X);

    arr = cobraml::core::Brarray(cobraml::core::Device::GPU, cobraml::core::Dtype::INT8, {10});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::GPU);
}


TEST(ArrayTestFunctionals, test_shape) {
    cobraml::core::Brarray arr(cobraml::core::Device::CPU, cobraml::core::Dtype::INT32, {10});
    ASSERT_EQ(arr.get_shape(), std::vector<size_t>{10});

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {1, 10, 10, 2});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {1, 10});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);

    ASSERT_THROW(cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {0}),
                 std::runtime_error);

    ASSERT_THROW(cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {1, 0, 1}),
                 std::runtime_error);

    ASSERT_THROW(cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {}),
                 std::runtime_error);
}

TEST(ArrayTestFunctionals, test_stride) {
    cobraml::core::Brarray arr(cobraml::core::Device::CPU, cobraml::core::Dtype::INT32, {10});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>{1});

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {7, 2, 8, 1});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({512, 256, 32, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT16, {4, 2, 3});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({32, 16, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT32, {2, 8, 8});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({64, 8, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT32, {1, 1, 1});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({8, 8, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::INT64, {5, 4});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({4, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32, {20, 3, 9});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({48, 16, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32, {1, 16});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({16, 1}));

    arr = cobraml::core::Brarray(cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT64, {10, 10, 10});
    ASSERT_EQ(arr.get_stride(), std::vector<size_t>({120, 12, 1}));
}

TEST(ArrayTestFunctionals, from_vector) {
    const std::vector vec{1, 2, 3, 4, 5, 6};

    std::vector<size_t> shape{6};
    cobraml::core::Brarray arr(cobraml::core::CPU, cobraml::core::INT32, shape, vec);
    ASSERT_EQ(arr.get_shape(), shape);
    for (size_t i{0}; i < 6; ++i)
        ASSERT_EQ(arr.get_buffer<int>()[i], vec[i]);

    shape = {2, 3, 1};
    arr = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT32, shape, vec);
    ASSERT_EQ(arr.get_shape(), shape);
    for (size_t i{0}; i < 6; ++i)
        ASSERT_EQ(arr.get_buffer<int>()[i * 8], vec[i]);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT32, {1, 3, 4}, vec),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, shape, vec),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, {0}, std::vector<int8_t>()),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, {}, vec),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, {6, 1, 0}, vec),
        std::runtime_error);

    ASSERT_THROW(
        cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, {0}, vec),
        std::runtime_error);
}

bool check_alignment(const void * ptr) {
#ifdef AVX2
    return reinterpret_cast<uintptr_t>(ptr) % 32 == 0;
#else
    return reinterpret_cast<uintptr_t>(ptr) % 8 == 0;
#endif
}

TEST(ArrayTestFunctionals, test_indexing) {
    const std::vector vec{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    const std::vector<size_t> shape_ten{2, 3, 4};
    const std::vector<size_t> shape_mat{3, 4};
    const std::vector<size_t> shape_vec{4};
    const std::vector<size_t> shape_scalar{1};

    const cobraml::core::Brarray arr(cobraml::core::CPU, cobraml::core::INT32, shape_ten, vec);
    ASSERT_TRUE(check_alignment(arr.get_buffer<int>()));

    for (size_t i{0}; i < shape_ten[0]; ++i) {
        cobraml::core::Brarray matrix{arr[i]};
        ASSERT_TRUE(check_alignment(arr.get_buffer<int>()));
        ASSERT_EQ(shape_mat, matrix.get_shape());
        for (size_t j{0}; j < shape_ten[1]; ++j) {
            cobraml::core::Brarray vector{matrix[j]};
            ASSERT_TRUE(check_alignment(vector.get_buffer<int>()));
            ASSERT_EQ(shape_vec, vector.get_shape());
            for (size_t k{0}; k < shape_ten[2]; ++k) {
                cobraml::core::Brarray scalar{vector[k]};
                ASSERT_EQ(shape_scalar, scalar.get_shape());
                ASSERT_EQ(scalar.item<int>(), vec[i * 12 + j * 4 + k]);
                ASSERT_TRUE(check_alignment(scalar.get_buffer<int>()));
            }
        }
    }

    ASSERT_THROW(arr[2], std::out_of_range);
    ASSERT_THROW(arr[0][3], std::out_of_range);
    ASSERT_THROW(arr[0][0][4], std::out_of_range);
    cobraml::core::Brarray arr2;
    ASSERT_THROW(arr2[0], std::runtime_error);

    cobraml::core::Brarray scal_arr{arr[0][0][0]};
    ASSERT_THROW(scal_arr[1], std::out_of_range);

    arr2 = arr[1][0];;
    arr2.get_buffer<int>()[2] = 20;
    ASSERT_EQ(arr[1][0][2].item<int>(), 20);

    // ensure scalars are deep copies
    scal_arr = arr[0][0][0];
    scal_arr.get_buffer<int>()[0] = 100;
    ASSERT_EQ(scal_arr[0].item<int>(), 100);
    ASSERT_NE(arr[0][0][0].item<int>(), 100);

    cobraml::core::Brarray gpu_tensor(cobraml::core::GPU, cobraml::core::FLOAT32, {10, 10});

    gpu_tensor = gpu_tensor[4];
    ASSERT_EQ(gpu_tensor.get_dtype(), cobraml::core::FLOAT32);
    ASSERT_EQ(gpu_tensor.get_device(), cobraml::core::GPU);

    gpu_tensor = gpu_tensor[0];
    ASSERT_EQ(gpu_tensor.get_dtype(), cobraml::core::FLOAT32);
    ASSERT_EQ(gpu_tensor.get_device(), cobraml::core::GPU);
}

TEST(ArrayTestFunctionals, default_constructor) {
    const cobraml::core::Brarray arr;
    ASSERT_TRUE(arr.get_shape().empty());
    ASSERT_EQ(arr.get_dtype(), cobraml::core::INVALID);
    ASSERT_TRUE(arr.get_stride().empty());
    ASSERT_EQ(arr.get_device(), cobraml::core::CPU);
}

TEST(ArrayTestFunctionals, test_copy_constructor) {
    std::vector vec{1, 2, 3, 4, 5, 6};
    const cobraml::core::Brarray arr(cobraml::core::CPU, cobraml::core::INT32, {6}, vec);
    auto arr2 = arr;

    ASSERT_EQ(arr.get_shape(), arr2.get_shape());
    ASSERT_EQ(arr.get_dtype(), arr2.get_dtype());
    ASSERT_EQ(arr.get_stride(), arr2.get_stride());
    ASSERT_EQ(arr.get_device(), arr2.get_device());

    const auto p1{arr.get_buffer<int>()};
    const auto p2{arr2.get_buffer<int>()};
    ASSERT_NE(p1, p2);

    for (size_t i{0}; i < arr.get_shape()[0]; ++i) {
        ASSERT_EQ(p1[i], p2[i]);
    }
}

TEST(ArrayTestFunctionals, test_copy_operator) {
    const std::vector vec{1, 2, 3, 4, 5, 6};
    const std::vector vec1{7, 8, 9, 10, 11, 12, 13};
    const std::vector vec2{7, 8};

    const cobraml::core::Brarray arr(cobraml::core::CPU, cobraml::core::INT32, {6}, vec);
    cobraml::core::Brarray arr1(cobraml::core::CPU, cobraml::core::INT32, {7}, vec1);
    cobraml::core::Brarray arr2(cobraml::core::CPU, cobraml::core::INT32, {2}, vec2);

    arr2 = arr;
    arr1 = arr;

    ASSERT_EQ(arr.get_shape(), arr2.get_shape());
    ASSERT_EQ(arr.get_dtype(), arr2.get_dtype());
    ASSERT_EQ(arr.get_stride(), arr2.get_stride());
    ASSERT_EQ(arr.get_device(), arr2.get_device());

    ASSERT_EQ(arr.get_shape(), arr1.get_shape());
    ASSERT_EQ(arr.get_dtype(), arr1.get_dtype());
    ASSERT_EQ(arr.get_stride(), arr1.get_stride());
    ASSERT_EQ(arr.get_device(), arr1.get_device());

    const auto p1{arr.get_buffer<int>()};
    const auto p2{arr2.get_buffer<int>()};
    const auto p3{arr1.get_buffer<int>()};

    ASSERT_NE(p1, p2);
    ASSERT_NE(p1, p3);

    for (size_t i{0}; i < arr.get_shape()[0]; ++i) {
        ASSERT_EQ(p1[i], p2[i]);
        ASSERT_EQ(p1[i], p3[i]);
    }
}

TEST(ArrayTestFunctionals, test_invalid_get_buffer) {
    cobraml::core::Brarray arr;
    ASSERT_THROW(arr.get_buffer<int>(), std::runtime_error);
    arr = cobraml::core::Brarray(cobraml::core::CPU, cobraml::core::INT8, {10, 10});
    ASSERT_THROW(arr.get_buffer<int>(), std::runtime_error);
}

TEST(ArrayTestFunctionals, print) {
    const std::vector<int8_t> vec{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    const cobraml::core::Brarray arr(cobraml::core::CPU, cobraml::core::INT8, {2, 3, 4}, vec);
    std::cout << arr << std::endl;
}

//modify and test gemv
