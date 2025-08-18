//
// Created by sriram on 8/17/25.
//

#include <benchmark/benchmark.h>
#include <random>
#include <cublas_v2.h>
#include <iostream>
#include <cute/numeric/numeric_types.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "kernels/ampere/gemm.cuh"

namespace {
    int one_to_10() {
        static std::mt19937 generator(std::random_device{}());
        static std::uniform_int_distribution distribution(0, 9);
        return distribution(generator);
    }

    using Half_T = cute::half_t;

    constexpr float options_f32[]{
        -3.1415926535, 2.7182818284, 1.6180339887, -0.5772156649, 4.6692016091, 1.4142135623, 2.3025850929,
        0.6931471806, 1.7320508075, -6.2831853071
    };

    template<
        typename VectorDtype,
        typename OptionsDtype
    >
    void fill_vector(thrust::host_vector<VectorDtype> &vec, const OptionsDtype &options) {
        for (VectorDtype &item: vec)
            item = Half_T(options[one_to_10()]);
    }

    void benchmark_cublas_tn(benchmark::State &st) {
        int const m{static_cast<int>(st.range(0))};
        int const n{static_cast<int>(st.range(1))};
        int const k{static_cast<int>(st.range(2))};

        thrust::host_vector<Half_T> host_a(m * k);
        thrust::host_vector<Half_T> host_b(k * n);
        thrust::host_vector<Half_T> host_c(m * n, Half_T(0));

        fill_vector(host_a, options_f32);
        fill_vector(host_b, options_f32);

        thrust::device_vector<Half_T> device_a{host_a};
        thrust::device_vector<Half_T> device_b{host_b};
        thrust::device_vector<Half_T> device_c{host_c};

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

        const Half_T alpha{Half_T(1.0f)};
        const Half_T beta{Half_T(0.0f)};

        for (auto _: st) {
            cublasGemmEx(
                handle,
                CUBLAS_OP_T, // op(B): No transpose (B is already column-major)
                CUBLAS_OP_N, // op(A): Transpose (A is row-major, need to transpose)
                n, // Number of rows in op(B) and C
                m, // Number of columns in op(A) and C
                k, // Number of columns in op(B) and rows in op(A)
                &alpha, // Alpha scalar
                thrust::raw_pointer_cast(device_b.data()), // Matrix B
                CUDA_R_16F, // Data type of B
                k, // Leading dimension of B
                thrust::raw_pointer_cast(device_a.data()), // Matrix A
                CUDA_R_16F, // Data type of A
                k, // Leading dimension of A
                &beta, // Beta scalar
                thrust::raw_pointer_cast(device_c.data()), // Matrix C
                CUDA_R_16F, // Data type of C
                m, // Leading dimension of C
                CUBLAS_COMPUTE_16F, // Compute type
                CUBLAS_GEMM_DEFAULT_TENSOR_OP // Algorithm
            );
        }
    }

    void benchmark_ampere(benchmark::State &st) {
        int const m{static_cast<int>(st.range(0))};
        int const n{static_cast<int>(st.range(1))};
        int const k{static_cast<int>(st.range(2))};

        thrust::host_vector<Half_T> host_a(m * k);
        thrust::host_vector<Half_T> host_b(k * n);
        thrust::host_vector<Half_T> host_c(m * n, Half_T(0));

        fill_vector(host_a, options_f32);
        fill_vector(host_b, options_f32);

        thrust::device_vector<Half_T> device_a{host_a};
        thrust::device_vector<Half_T> device_b{host_b};
        thrust::device_vector<Half_T> device_c{host_c};

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

        for (auto _: st) {
            cobraml::runner_tn(
                m,
                n,
                k,
                Half_T(1),
                thrust::raw_pointer_cast(device_a.data()),
                k,
                thrust::raw_pointer_cast(device_b.data()),
                k,
                Half_T(0),
                thrust::raw_pointer_cast(device_c.data()),
                m
            );
        }
    }

    BENCHMARK(benchmark_cublas_tn)
    ->Args({1024, 1024, 1024})
    ->Threads(1);

    BENCHMARK(benchmark_ampere)
    ->Args({1024, 1024, 1024})
    ->Threads(1);
}

BENCHMARK_MAIN();
