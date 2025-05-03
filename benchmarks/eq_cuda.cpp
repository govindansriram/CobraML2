//
// Created by sriram on 5/3/25.
//

#include <benchmark/benchmark.h>
#include <random>
#include "enums.h"
#include "brarray.h"

void cuda_eq(benchmark::State &st) {
    size_t const rows{static_cast<size_t>(st.range(0))};
    size_t const cols{static_cast<size_t>(st.range(1))};
    size_t const type{static_cast<size_t>(st.range(2))};

    cobraml::core::func_pos = type;

    const cobraml::core::Brarray res(cobraml::core::CUDA, cobraml::core::INT32, {rows, cols});
    const cobraml::core::Brarray res2(cobraml::core::CUDA, cobraml::core::INT32, {rows, cols});

    for (auto _: st) {
        bool state{res == res2};
        benchmark::DoNotOptimize(state);
    }

    st.counters["rows"] = rows;
    st.counters["columns"] = cols;
    st.counters["type"] = cobraml::core::func_pos;
}

BENCHMARK(cuda_eq)
->Args({8000, 8000, 0})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({8000, 8000, 1})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({8000, 8000, 2})
->Threads(1);

BENCHMARK_MAIN();
