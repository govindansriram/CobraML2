//
// Created by sriram on 12/19/24.
//

#include <benchmark/benchmark.h>
#include <random>
#include "matrix.h"

namespace {
    class CPUFixture : public benchmark::Fixture {
    public:
        double lower_bound{0};
        double upper_bound{10000};
        std::uniform_real_distribution<> unif{lower_bound, upper_bound};
        std::default_random_engine gen{108};

        std::vector<std::vector<double> > create_vector(size_t const rows, size_t const columns) {
            std::vector ret(rows, std::vector(columns, 0.0));

            for (auto &vector: ret) {
                for (auto &num: vector) {
                    num = unif(gen);
                }
            }

            return ret;
        }
    };

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductAll)(benchmark::State &st) {
        size_t const rows{static_cast<size_t>(st.range(0))};
        size_t const col{static_cast<size_t>(st.range(1))};
        size_t const pos{static_cast<size_t>(st.range(2))};

        cobraml::core::func_pos = pos;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(rows, col), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, col), cobraml::core::CPU);

        cobraml::core::Matrix res(1, rows, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
            benchmark::DoNotOptimize(res);
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = col;
        st.counters["type"] = pos;
    }

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductAll)
    // ->Args({5000, 5000, 0})
    // ->Args({5000, 5000, 1})
    ->Args({5000, 5000, 2})
    ->Args({5000, 5000, 3})
    ->Threads(1);
}

BENCHMARK_MAIN();
