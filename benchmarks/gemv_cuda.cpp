//
// Created by sriram on 5/10/25.
//

#include <iostream>
#include <benchmark/benchmark.h>
#include <random>
#include "enums.h"
#include "brarray.h"

void cuda_gemv(benchmark::State &st) {
    size_t length{static_cast<size_t>(st.range(0))};
    length  = 1 << length;
    size_t const type{static_cast<size_t>(st.range(1))};

    cobraml::core::func_pos = type;

    const cobraml::core::Brarray mat(cobraml::core::CUDA, cobraml::core::FLOAT32, {length, length});
    const cobraml::core::Brarray vec(cobraml::core::CUDA, cobraml::core::FLOAT32, {length});

    for (auto _: st) {
        auto final{cobraml::core::gemv(mat, vec, 1.0f, 1.0f)};
        benchmark::DoNotOptimize(final);
    }

    st.counters["length"] = length;
    st.counters["type"] = cobraml::core::func_pos;
}

BENCHMARK(cuda_gemv)
->Args({12, 0})
->Threads(1);

BENCHMARK(cuda_gemv)
->Args({12, 1})
->Threads(1);

BENCHMARK(cuda_gemv)
->Args({12, 2})
->Threads(1);

BENCHMARK(cuda_gemv)
->Args({12, 3})
->Threads(1);

BENCHMARK(cuda_gemv)
->Args({12, 4})
->Threads(1);

BENCHMARK(cuda_gemv)
->Args({12, 5})
->Threads(1);


// BENCHMARK(cuda_gemm)
// ->Args({12, 1})
// ->Threads(1);
//
// BENCHMARK(cuda_gemm)
// ->Args({12, 2})
// ->Threads(1);
//
// BENCHMARK(cuda_gemm)
// ->Args({12, 3})
// ->Threads(1);
//
// BENCHMARK(cuda_gemm)
// ->Args({12, 4})
// ->Threads(1);

BENCHMARK_MAIN();
