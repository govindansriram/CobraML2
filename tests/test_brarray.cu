//
// Created by sriram on 1/26/25.
//
#include <random>
#include <gtest/gtest.h>
#include "brarray.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cute/tensor.hpp>
#include "kernels/ampere/gemm.cuh"
#include "kernels/naive_matmul.cuh"

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

template<typename Dtype>
void init(
    thrust::host_vector<Dtype> &vector,
    const int x,
    const int y,
    const bool row_major=true,
    const int start=0) {

    // std::random_device rd;
    std::mt19937 gen(108);

    std::uniform_int_distribution distrib_bottom(1, 10);
    std::uniform_int_distribution distrib_top(0, 9);

    for (int i{0}; i < x; ++i) {
        for (int j{0}; j < y; ++j) {
            const auto top{static_cast<float>(distrib_top(gen))};
            const auto bottom{static_cast<float>(distrib_bottom(gen))};

            Dtype val{static_cast<Dtype>(top / bottom)};

            int pos{row_major ?  i * y + j : i + j * x};
            vector[pos] = val;
        }
    }

}

template<typename DType>
bool approx_equal(DType a, DType b, float epsilon = 1e-2f) {
    float rel_err = std::abs(a - b) / std::max(std::abs(b), 1e-6f);
    return rel_err < epsilon;
}

TEST(BrarrayTest, visualize) {

    const int m{512};
    const int n{1024};
    const int k{256};

    thrust::host_vector<half_t> host_a(m * k);
    thrust::host_vector<half_t> host_b(k * n);
    thrust::host_vector<half_t> host_c(m * n, half_t(0));
    thrust::host_vector<half_t> host_c_ref(m * n, half_t(0));

    init(host_a, m, k);
    init(host_b, k, n, false);
    init(host_c, m, n, false);
    init(host_c_ref, m, n, false);

    thrust::device_vector<half_t> device_a{host_a};
    thrust::device_vector<half_t> device_b{host_b};
    thrust::device_vector<half_t> device_c{host_c};
    thrust::device_vector<half_t> device_c_ref{host_c_ref};

    runner_tn(
        m,
        n,
        k,
        half_t(1),
        thrust::raw_pointer_cast(device_a.data()),
        k,
        thrust::raw_pointer_cast(device_b.data()),
        k,
        half_t(0),
        thrust::raw_pointer_cast(device_c.data()),
        m
    );

    thrust::copy(device_c.begin(), device_c.end(), host_c.begin());
    naive_matmul_TNN<half_t, half_t><<<1, 1>>>(
        m,
        n,
        k,
        thrust::raw_pointer_cast(device_a.data()),
        thrust::raw_pointer_cast(device_b.data()),
        thrust::raw_pointer_cast(device_c_ref.data()));

    thrust::copy(device_c_ref.begin(), device_c_ref.end(), host_c_ref.begin());

    for (size_t i{0}; i < m * n; ++i) {
        ASSERT_TRUE(approx_equal(host_c[i], host_c_ref[i]));
    }
    CUTE_CHECK_LAST();
}

//modify and test gemv
