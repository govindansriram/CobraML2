//
// Created by sriram on 12/19/24.
//

#include <benchmark/benchmark.h>
#include <random>
#include "matrix.h"

namespace {
    int one_to_10() {
        static std::mt19937 generator(std::random_device{}());
        static std::uniform_int_distribution distribution(0, 9);
        return distribution(generator);
    }

    constexpr double optionsf64[]{
        -3.1415926535, 2.7182818284, 1.6180339887, -0.5772156649, 4.6692016091, 1.4142135623, 2.3025850929,
        0.6931471806, 1.7320508075, -6.2831853071
    };

    constexpr float optionsf32[]{
        -3.1415926535, 2.7182818284, 1.6180339887, -0.5772156649, 4.6692016091, 1.4142135623, 2.3025850929,
        0.6931471806, 1.7320508075, -6.2831853071
    };

#define FILL(option, matrix, rows, columns) {\
    for(size_t i = 0; i < rows; ++i){\
        for(size_t j = 0; j < columns; ++j){\
            matrix[i][j] = option[one_to_10()];\
        }\
    }\
}

    void cpu_gemv_f64_runner(benchmark::State &st) {
        size_t const rows{static_cast<size_t>(st.range(0))};
        size_t const col{static_cast<size_t>(st.range(1))};

        std::vector matrix_vec(rows, std::vector(col, 0.0));
        std::vector vector_vec(1, std::vector(col, 0.0));
        cobraml::core::Matrix res(1, rows, cobraml::core::CPU, cobraml::core::FLOAT64);

        FILL(optionsf64, matrix_vec, rows, col);
        FILL(optionsf64, vector_vec, 1, col);

        constexpr double a{1.567};
        constexpr double b{2.3987};

        cobraml::core::Matrix mat{from_vector(matrix_vec, cobraml::core::CPU)};
        cobraml::core::Matrix vec{from_vector(vector_vec, cobraml::core::CPU)};

        for (auto _: st) {
            gemv(mat, vec, res, a, b);
            benchmark::DoNotOptimize(res);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = col;
        st.counters["type"] = cobraml::core::func_pos;
    }

    void cpu_gemv_f32_runner(benchmark::State &st) {
        size_t const rows{static_cast<size_t>(st.range(0))};
        size_t const col{static_cast<size_t>(st.range(1))};

        std::vector matrix_vec(rows, std::vector(col, 0.0f));
        std::vector vector_vec(1, std::vector(col, 0.0f));
        cobraml::core::Matrix res(1, rows, cobraml::core::CPU, cobraml::core::FLOAT32);

        FILL(optionsf32, matrix_vec, rows, col);
        FILL(optionsf32, vector_vec, 1, col);

        constexpr float a{1.567};
        constexpr float b{2.3987};

        cobraml::core::Matrix mat{from_vector(matrix_vec, cobraml::core::CPU)};
        cobraml::core::Matrix vec{from_vector(vector_vec, cobraml::core::CPU)};

        for (auto _: st) {
            gemv(mat, vec, res, a, b);
            benchmark::DoNotOptimize(res);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = col;
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
