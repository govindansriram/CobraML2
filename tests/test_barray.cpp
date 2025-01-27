//
// Created by sriram on 1/26/25.
//
#include <gtest/gtest.h>
#include "barray.h"

TEST(ArrayTestFunctionals, test_dtype) {
    cobraml::core::Array arr(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INT8);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT8);

    arr = cobraml::core::Array(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INT16);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT16);

    arr = cobraml::core::Array(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT32);

    arr = cobraml::core::Array(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INT64);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INT64);

    arr = cobraml::core::Array(10, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::FLOAT32);

    arr = cobraml::core::Array(10, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT64);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::FLOAT64);

    ASSERT_THROW(
        cobraml::core::Array(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INVALID),
        std::runtime_error);
}

TEST(ArrayTestFunctionals, test_device) {
    cobraml::core::Array arr(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INT8);
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);

    arr = cobraml::core::Array(10, cobraml::core::Device::CPU_X, cobraml::core::Dtype::INT16);
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU_X);

    arr = cobraml::core::Array(10, cobraml::core::Device::GPU, cobraml::core::Dtype::INT32);
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::GPU);
}

TEST(ArrayTestFunctionals, test_len) {
    cobraml::core::Array arr(10, cobraml::core::Device::CPU, cobraml::core::Dtype::INT8);
    ASSERT_EQ(arr.len(), 10);

    arr = cobraml::core::Array(100, cobraml::core::Device::CPU_X, cobraml::core::Dtype::INT16);
    ASSERT_EQ(arr.len(), 100);

    arr = cobraml::core::Array(24, cobraml::core::Device::GPU, cobraml::core::Dtype::INT32);
    ASSERT_EQ(arr.len(), 24);
}

TEST(ArrayTestFunctionals, test_default_constructor) {
    cobraml::core::Array const arr;
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);
    ASSERT_EQ(arr.get_dtype(), cobraml::core::Dtype::INVALID);
    ASSERT_EQ(arr.len(), 0);
}

#define CHECK_EQUAL(pointer_1, pointer_2, length, ref) \
    for(size_t i = 0; i < length; ++i){\
        if(pointer_1[i] != pointer_2[i]){\
            ref = false;\
        }\
    }\
    ref = true;\

TEST(ArrayTestFunctionals, from_vector) {
    std::vector const vec{1, 2, 3, 4, 5, 6};
    std::vector const vec2{1.5f, 2.22f, 3.33f, 4.26f, 5.12f, 6.0f};

    const cobraml::core::Array i_arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    const cobraml::core::Array f_arr{from_vector(vec2, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32)};

    ASSERT_EQ(i_arr.len(), vec.size());
    ASSERT_EQ(i_arr.get_device(), cobraml::core::CPU);
    ASSERT_EQ(i_arr.get_dtype(), cobraml::core::INT32);

    ASSERT_EQ(f_arr.len(), vec2.size());
    ASSERT_EQ(f_arr.get_device(), cobraml::core::CPU);
    ASSERT_EQ(f_arr.get_dtype(), cobraml::core::FLOAT32);
    ASSERT_THROW(from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT8), std::runtime_error);

    const auto i_buff = cobraml::core::get_buffer<int>(i_arr);
    const auto f_buff = cobraml::core::get_buffer<float>(f_arr);
    bool flag;

    CHECK_EQUAL(vec.data(), i_buff, vec.size(), flag);
    ASSERT_EQ(flag, true);

    CHECK_EQUAL(vec2.data(), f_buff, vec2.size(), flag);
    ASSERT_EQ(flag, true);
}

TEST(ArrayTestFunctionals, test_indexing) {
    std::vector const vec{0, 1, 2, 3, 4, 5};
    const cobraml::core::Array arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};

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
    const cobraml::core::Array arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    cobraml::core::Array arr2 = arr;

    arr2[2].set_item(10);

    ASSERT_EQ(arr2[2].item<int>(), 10);
}

TEST(ArrayTestFunctionals, test_copy_constructor) {
    std::vector const vec{0, 1, 2, 3, 4, 5};
    const cobraml::core::Array arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    cobraml::core::Array arr2 = arr;

    arr[0].set_item(7);

    const auto buff_1 = cobraml::core::get_buffer<int>(arr);
    const auto buff_2 = cobraml::core::get_buffer<int>(arr2);
    bool flag;

    CHECK_EQUAL(buff_1, buff_2, vec.size(), flag);
    ASSERT_EQ(flag, true);
}

TEST(ArrayTestFunctionals, test_copy_assigment) {
    std::vector const vec{0, 1, 2, 3, 4, 5};
    std::vector const vec2{1.5f, 2.22f, 3.33f, 4.26f, 5.12f, 6.0f, 7.0f};
    cobraml::core::Array arr{from_vector(vec, cobraml::core::Device::CPU, cobraml::core::Dtype::INT32)};
    const cobraml::core::Array arr_2{from_vector(vec2, cobraml::core::Device::CPU, cobraml::core::Dtype::FLOAT32)};

    arr = arr_2;

    arr_2[4].set_item(1.5f);

    const auto buff_1 = cobraml::core::get_buffer<float>(arr);
    const auto buff_2 = cobraml::core::get_buffer<float>(arr_2);
    bool flag;

    CHECK_EQUAL(buff_1, buff_2, vec.size(), flag);
    ASSERT_EQ(arr.len(), arr_2.len());
    ASSERT_EQ(arr.get_device(), arr_2.get_device());
    ASSERT_EQ(arr.get_dtype(), arr.get_dtype());
    ASSERT_EQ(flag, true);
}


