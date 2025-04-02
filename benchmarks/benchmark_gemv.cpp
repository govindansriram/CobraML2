//
// Created by sriram on 12/19/24.
//

#include <benchmark/benchmark.h>
#include <random>
#include "enums.h"
#include "brarray.h"

namespace {

    int one_to_10() {
        static std::mt19937 generator(std::random_device{}());
        static std::uniform_int_distribution distribution(0, 9);
        return distribution(generator);
    }

#define FILL_VECTOR(arr, vector) for(size_t i = 0; i < vector.size(); ++i) vector[i] = arr[one_to_10()];

    constexpr double optionsf64[]{
        -3.1415926535, 2.7182818284, 1.6180339887, -0.5772156649, 4.6692016091, 1.4142135623, 2.3025850929,
        0.6931471806, 1.7320508075, -6.2831853071
    };

    constexpr float optionsf32[]{
        -3.1415926535, 2.7182818284, 1.6180339887, -0.5772156649, 4.6692016091, 1.4142135623, 2.3025850929,
        0.6931471806, 1.7320508075, -6.2831853071
    };

    void cpu_gemv_f64_runner(benchmark::State &st) {
        size_t const rows{static_cast<size_t>(st.range(0))};
        size_t const cols{static_cast<size_t>(st.range(1))};

        std::vector matrix_vec(rows * cols, 0.0);
        std::vector vector_vec(cols, 0.0);

        FILL_VECTOR(optionsf64, matrix_vec);
        FILL_VECTOR(optionsf64, vector_vec);

        cobraml::core::Brarray res(cobraml::core::CPU, cobraml::core::FLOAT64, {rows});

        constexpr double a{1.567};
        constexpr double b{2.3987};

        cobraml::core::Brarray mat(cobraml::core::CPU, cobraml::core::FLOAT64, {rows, cols}, matrix_vec);
        const cobraml::core::Brarray vec(cobraml::core::CPU, cobraml::core::FLOAT64, {cols}, vector_vec);

        for (auto _: st) {
            gemv(res, mat, vec, a, b);
            benchmark::DoNotOptimize(res);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = cols;
        st.counters["type"] = cobraml::core::func_pos;
    }

    void cpu_gemv_f32_runner(benchmark::State &st) {
        size_t const rows{static_cast<size_t>(st.range(0))};
        size_t const cols{static_cast<size_t>(st.range(1))};

        std::vector matrix_vec(rows * cols, 0.0f);
        std::vector vector_vec(cols, 0.0f);

        FILL_VECTOR(optionsf32, matrix_vec);
        FILL_VECTOR(optionsf32, vector_vec);

        cobraml::core::Brarray res(cobraml::core::CPU, cobraml::core::FLOAT32, {rows});

        constexpr float a{1.567f};
        constexpr float b{2.3987f};

        cobraml::core::Brarray mat(cobraml::core::CPU, cobraml::core::FLOAT32, {rows, cols}, matrix_vec);
        const cobraml::core::Brarray vec(cobraml::core::CPU, cobraml::core::FLOAT32, {cols}, vector_vec);

        for (auto _: st) {
            gemv(res, mat, vec, a, b);
            benchmark::DoNotOptimize(res);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = cols;
        st.counters["type"] = cobraml::core::func_pos;
    }

    void cpu_gemv_f64_naive(benchmark::State &st) {
        cobraml::core::func_pos = 0;
        cpu_gemv_f64_runner(st);
    }

    void cpu_gemv_f64_parallel(benchmark::State &st) {
        cobraml::core::func_pos = 1;
        cpu_gemv_f64_runner(st);
    }

    void cpu_gemv_f64_parallel_simd(benchmark::State &st) {
        cobraml::core::func_pos = 2;
        cpu_gemv_f64_runner(st);
    }

    void cpu_gemv_f64_parallel_custom(benchmark::State &st) {
        cobraml::core::func_pos = 3;
        cpu_gemv_f64_runner(st);
    }

    void cpu_gemv_f32_naive(benchmark::State &st) {
        cobraml::core::func_pos = 0;
        cpu_gemv_f32_runner(st);
    }

    void cpu_gemv_f32_parallel(benchmark::State &st) {
        cobraml::core::func_pos = 1;
        cpu_gemv_f32_runner(st);
    }

    void cpu_gemv_f32_parallel_simd(benchmark::State &st) {
        cobraml::core::func_pos = 2;
        cpu_gemv_f32_runner(st);
    }

    void cpu_gemv_f32_parallel_custom(benchmark::State &st) {
        cobraml::core::func_pos = 3;
        cpu_gemv_f32_runner(st);
    }

    BENCHMARK(cpu_gemv_f64_naive)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f64_parallel)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f64_parallel_simd)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f64_parallel_custom)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f32_naive)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f32_parallel)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f32_parallel_simd)
    ->Args({5000, 5000})
    ->Threads(1);

    BENCHMARK(cpu_gemv_f32_parallel_custom)
    ->Args({5000, 5000})
    ->Threads(1);
}

BENCHMARK_MAIN();
