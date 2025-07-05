//
// Created by sriram on 1/26/25.
//
#include <random>
#include <gtest/gtest.h>
#include "brarray.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

//modify and test gemv
