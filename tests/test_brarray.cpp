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
            }
        }
    }

    ASSERT_THROW(arr[2], std::out_of_range);
    ASSERT_THROW(arr[0][3], std::out_of_range);
    ASSERT_THROW(arr[0][0][4], std::out_of_range);

    cobraml::core::Brarray arr2;
    ASSERT_THROW(arr2[0], std::runtime_error);

    // TODO test shallow copy
}

//TODO test copy constructor & printing then modify and test gemv

/**
TEST(ArrayTestFunctionals, test_default_constructor) {
    cobraml::core::Barray const arr;
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INVALID);
    ASSERT_EQ(arr.len(), 0);
}

#define CHECK_EQUAL(pointer_1, pointer_2, length, ref) {\
    for(size_t i = 0; i < length; ++i){\
        if(pointer_1[i] != pointer_2[i]){\
            ref = false;\
        }\
    }\
}

TEST(ArrayTestFunctionals, test_invalid_buffer) {
    cobraml::core::Barray const arr;
    ASSERT_THROW(cobraml::core::get_buffer<int8_t>(arr), std::runtime_error);
}

TEST(ArrayTestFunctionals, from_vector) {
    std::vector const vec{1, 2, 3, 4, 5, 6};
    std::vector const vec2{1.5f, 2.22f, 3.33f, 4.26f, 5.12f, 6.0f};
    std::vector<int8_t> vec3{};

    const cobraml::core::Barray i_arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    const cobraml::core::Barray f_arr{from_vector(vec2, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32)};

    ASSERT_EQ(i_arr.len(), vec.size());
    ASSERT_EQ(i_arr.get_device(), cobraml::core::CPU);
    ASSERT_EQ(i_arr.get_dtype(), cobraml::core::INT32);

    ASSERT_EQ(f_arr.len(), vec2.size());
    ASSERT_EQ(f_arr.get_device(), cobraml::core::CPU);
    ASSERT_EQ(f_arr.get_dtype(), cobraml::core::FLOAT32);
    ASSERT_THROW(from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT8), std::runtime_error);

    const auto i_buff = cobraml::core::get_buffer<int>(i_arr);
    const auto f_buff = cobraml::core::get_buffer<float>(f_arr);
    bool flag{true};

    CHECK_EQUAL(vec.data(), i_buff, vec.size(), flag);
    ASSERT_EQ(flag, true);

    CHECK_EQUAL(vec2.data(), f_buff, vec2.size(), flag);
    ASSERT_EQ(flag, true);

    ASSERT_THROW(from_vector(vec3, cobraml::core::Device::CPU, cobraml::core::Dtype::INT8), std::runtime_error);
}

TEST(ArrayTestFunctionals, test_indexing) {
    std::vector const vec{0, 1, 2, 3, 4, 5};
    const cobraml::core::Barray arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};

    ASSERT_THROW(arr[10], std::out_of_range);

    for (size_t i = 0; i < arr.len(); ++i) {
        ASSERT_EQ(arr[i].item<int>(), i);
    }

    for (size_t i = 0; i < 1; ++i) {
        ASSERT_THROW(arr[i].item<int8_t>(), std::runtime_error);
    }
}

TEST(ArrayTestFunctionals, set_item) {
    std::vector const vec{0, 1, 2};
    const cobraml::core::Barray arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    cobraml::core::Barray arr2 = arr;

    arr2[2].set_item(10);

    ASSERT_EQ(arr2[2].item<int>(), 10);
}

TEST(ArrayTestFunctionals, test_copy_constructor) {
    std::vector const vec{0, 1, 2, 3, 4, 5};
    const cobraml::core::Barray arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    cobraml::core::Barray arr2 = arr;

    arr[0].set_item(7);

    const auto buff_1 = cobraml::core::get_buffer<int>(arr);
    const auto buff_2 = cobraml::core::get_buffer<int>(arr2);
    bool flag{true};

    CHECK_EQUAL(buff_1, buff_2, vec.size(), flag);
    ASSERT_EQ(flag, true);
}

TEST(ArrayTestFunctionals, test_copy_assigment) {
    std::vector const vec{0, 1, 2, 3, 4, 5};
    std::vector const vec2{1.5f, 2.22f, 3.33f, 4.26f, 5.12f, 6.0f, 7.0f};
    cobraml::core::Barray arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    const cobraml::core::Barray arr_2{from_vector(vec2, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32)};

    arr = arr_2;

    arr_2[4].set_item(1.5f);

    const auto buff_1 = cobraml::core::get_buffer<float>(arr);
    const auto buff_2 = cobraml::core::get_buffer<float>(arr_2);
    bool flag{true};

    CHECK_EQUAL(buff_1, buff_2, vec.size(), flag);
    ASSERT_EQ(arr.len(), arr_2.len());
    ASSERT_EQ(arr.get_device(), arr_2.get_device());
    ASSERT_EQ(arr.get_dtype(), arr.get_dtype());
    ASSERT_EQ(flag, true);
}

TEST(ArrayTestFunctionals, test_deep_copy) {
    std::vector const vec{1.5f, 2.22f, 3.33f, 4.26f, 5.12f, 6.0f, 7.0f};
    const cobraml::core::Barray arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32)};

    const cobraml::core::Barray arr_2{arr.deep_copy()};
    arr[6].set_item(8.8f);

    for (size_t i = 0; i < arr.len() - 1; ++i) {
        ASSERT_EQ(arr[i].item<float>(), arr_2[i].item<float>());
    }

    ASSERT_NE(arr[vec.size() - 1].item<float>(), arr_2[vec.size() - 1].item<float>());

    const cobraml::core::Barray arr_3;

    ASSERT_THROW(auto const arr_4 = arr_3.deep_copy(), std::runtime_error);

    auto const arr_5 = arr[3];

    ASSERT_EQ(arr_5.item<float>(), arr[3].item<float>());
}


TEST(ArrayTestFunctionals, test_print) {
    std::vector const vec{1.52345f, 2.2289761f, 0.0000333f, 4.26f, 1231235.1222f, 6.0000001f, 0.0f};
    const cobraml::core::Barray arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32)};
    std::cout << arr;

    std::cout << arr[5];

    const cobraml::core::Barray arr_2;
    ASSERT_THROW(std::cout << arr_2, std::runtime_error);

    cobraml::core::Barray const b(40, cobraml::core::CPU, cobraml::core::INT8);
    std::cout << b;
}
**/
