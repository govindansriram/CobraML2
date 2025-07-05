//
// Created by sriram on 5/3/25.
//

#include <benchmark/benchmark.h>
#include <random>
#include "enums.h"
#include "brarray.cuh"

void cuda_eq(benchmark::State &st) {
    size_t length{static_cast<size_t>(st.range(0))};
    length  = 1 << length;
    size_t const type{static_cast<size_t>(st.range(1))};

    cobraml::core::func_pos = type;

    const cobraml::core::Brarray res(cobraml::core::CUDA, cobraml::core::INT32, {length});
    const cobraml::core::Brarray res2(cobraml::core::CUDA, cobraml::core::INT32, {length});

    for (auto _: st) {
        bool state{res == res2};
        benchmark::DoNotOptimize(state);
    }

    st.counters["length"] = length;
    st.counters["type"] = cobraml::core::func_pos;
}

BENCHMARK(cuda_eq)
->Args({22, 0})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({22, 1})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({22, 2})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({22, 3})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({22, 4})
->Threads(1);

BENCHMARK(cuda_eq)
->Args({22, 5})
->Threads(1);

BENCHMARK_MAIN();
