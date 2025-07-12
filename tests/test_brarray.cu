//
// Created by sriram on 1/26/25.
//
#include <random>
#include <gtest/gtest.h>
#include "brarray.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cute/tensor.hpp>
#include "kernels/lovelace/gemm.cuh"

using namespace cobraml;

TEST(BrarrayTest, constructor_from_device_vector) {

    const Layout dynamic_layout_right{make_layout(make_shape(8, 4), make_stride(8, 1))};
    constexpr Layout<Shape<_3, _4>, Stride<_4, _1>> static_layout_right{};
    const Layout high_dimensional_layout{make_layout(make_shape(5, make_shape(7, 3)), LayoutLeft{})};

    const thrust::host_vector<float> host_vector_1(64);
    const thrust::host_vector<float> host_vector_2(12);
    const thrust::host_vector<float> host_vector_3(1);
    const thrust::host_vector<float> host_vector_4(105);

    {
        thrust::device_vector<float> device_vector_pass{host_vector_1};
        thrust::device_vector<float> device_vector_fail{host_vector_3};

        ASSERT_NE(reinterpret_cast<uintptr_t>(device_vector_fail.data().get()), reinterpret_cast<uintptr_t>(nullptr));
        ASSERT_THROW(Brarray(dynamic_layout_right, device_vector_fail), std::runtime_error);
        ASSERT_NO_THROW(Brarray(dynamic_layout_right, device_vector_pass));
        ASSERT_EQ(reinterpret_cast<uintptr_t>(device_vector_fail.data().get()), reinterpret_cast<uintptr_t>(nullptr));
    }

    {
        thrust::device_vector<float> device_vector_pass{host_vector_2};
        thrust::device_vector<float> device_vector_fail{host_vector_3};
        ASSERT_THROW(Brarray(static_layout_right, device_vector_fail), std::runtime_error);
        ASSERT_NO_THROW(Brarray(static_layout_right, device_vector_pass));
    }

    {
        thrust::device_vector<float> device_vector_pass{host_vector_4};
        thrust::device_vector<float> device_vector_fail{host_vector_3};
        ASSERT_THROW(Brarray(high_dimensional_layout, device_vector_fail), std::runtime_error);
        ASSERT_NO_THROW(Brarray(high_dimensional_layout, device_vector_pass));
    }
}

TEST(BrarrayTest, constructor_from_layout) {

    const Layout dynamic_layout_right{make_layout(make_shape(8, 4), make_stride(8, 1))};
    constexpr Layout<Shape<_3, _4>, Stride<_4, _2>> static_layout_right{};
    const Layout high_dimensional_layout{make_layout(make_shape(5, make_shape(9, 3)), LayoutLeft{})};

    {
        Brarray br(high_dimensional_layout, 11.01f);
        ASSERT_EQ(br.to_host().size(), 135);
        for (float &num: br.to_host())
            ASSERT_EQ(num, 11.01f);
    }

    {
        Brarray br(static_layout_right, 108.01f);
        ASSERT_EQ(br.to_host().size(), 15);
        for (float &num: br.to_host())
            ASSERT_EQ(num, 108.01f);
    }

    {
        Brarray br(dynamic_layout_right, 99);
        ASSERT_EQ(br.to_host().size(), 60);
        for (int &num: br.to_host())
            ASSERT_EQ(num, 99);
    }
}

TEST(BrarrayTest, constructor_from_vector) {
    constexpr Layout<Shape<_3, _4>, Stride<_4, _1>> static_layout_right{};

    {
        const std::vector<float> buffer{
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11
        };

        Brarray br(static_layout_right, buffer);

        auto host_buff{br.to_host()};

        for (size_t i{0}; i < buffer.size(); ++i)
            ASSERT_EQ(host_buff[i], buffer[i]);
    }

    {
        const std::vector<float> buffer{
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10
        };

        ASSERT_THROW(Brarray br(static_layout_right, buffer), std::runtime_error);
    }
}

TEST(BrarrayTest, visualize) {
    constexpr Layout<Shape<_512, _256>, Stride<_256, _1>> layout_1{};
    constexpr Layout<Shape<_1024, _256>, Stride<_256, _1>> layout_2{};
    constexpr Layout<Shape<_512, _1024>, Stride<_1024, _1>> layout_3{};

    const Brarray a(layout_1, half_t(108.1f));
    const Brarray b(layout_2, half_t(9.9f));
    Brarray c(layout_3, half_t(10.01f));

    runner_tn(a, b, c);
}

//modify and test gemv
