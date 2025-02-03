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

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductVtuneBig2)(benchmark::State &st){

        cobraml::core::func_pos = 2;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(10015, 10015), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, 10015), cobraml::core::CPU);

        cobraml::core::Matrix res(1, 10015, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
        }
    }

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductVtuneSmall2)(benchmark::State &st){

        cobraml::core::func_pos = 2;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(2515, 2515), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, 2515), cobraml::core::CPU);

        cobraml::core::Matrix res(1, 2515, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
        }
    }

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductVtuneBig3)(benchmark::State &st){

        cobraml::core::func_pos = 3;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(10015, 10015), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, 10015), cobraml::core::CPU);

        cobraml::core::Matrix res(1, 10015, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
        }
    }

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductVtuneSmall3)(benchmark::State &st){

        cobraml::core::func_pos = 3;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(2515, 2515), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, 2515), cobraml::core::CPU);

        cobraml::core::Matrix res(1, 2515, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
        }
    }

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductVtuneBig4)(benchmark::State &st){

        cobraml::core::func_pos = 4;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(10015, 10015), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, 10015), cobraml::core::CPU);

        cobraml::core::Matrix res(1, 10015, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
        }
    }

    BENCHMARK_DEFINE_F(CPUFixture, BatchedDotProductVtuneSmall4)(benchmark::State &st){

        cobraml::core::func_pos = 4;

        cobraml::core::Matrix const mat = from_vector(
            create_vector(2515, 2515), cobraml::core::CPU);

        cobraml::core::Matrix const vec = from_vector(
            create_vector(1, 2515), cobraml::core::CPU);

        cobraml::core::Matrix res(1, 2515, cobraml::core::CPU, cobraml::core::FLOAT64);

        constexpr double alpha1{1};

        for (auto _: st) {
            gemv(mat, vec, res, alpha1, alpha1);
        }
    }

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
        }

        st.counters["rows"] = rows;
        st.counters["columns"] = col;
        st.counters["type"] = pos;
    }

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductVtuneBig2)
    ->Threads(1);

    // BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductVtuneBig3)
    // ->Threads(1);

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductVtuneBig4)
    ->Threads(1);

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductVtuneSmall2)
    ->Threads(1);
    //
    // BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductVtuneSmall3)
    // ->Threads(1);

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductVtuneSmall4)
    ->Threads(1);

    BENCHMARK_REGISTER_F(CPUFixture, BatchedDotProductAll)
    ->Args({100, 100, 0})
    ->Args({200, 200, 0})
    ->Args({300, 300, 0})
    ->Args({400, 400, 0})
    ->Args({500, 500, 0})
    ->Args({600, 600, 0})
    ->Args({700, 700, 0})
    ->Args({800, 800, 0})
    ->Args({900, 900, 0})
    ->Args({1000, 1000, 0})
    ->Args({1100, 1100, 0})
    ->Args({1200, 1200, 0})
    ->Args({1300, 1300, 0})
    ->Args({1400, 1400, 0})
    ->Args({1500, 1500, 0})
    ->Args({1600, 1600, 0})
    ->Args({1700, 1700, 0})
    ->Args({1800, 1800, 0})
    ->Args({1900, 1900, 0})
    ->Args({2000, 2000, 0})
    ->Args({2100, 2100, 0})
    ->Args({2200, 2200, 0})
    ->Args({2300, 2300, 0})
    ->Args({2400, 2400, 0})
    ->Args({2500, 2500, 0})
    ->Args({2600, 2600, 0})
    ->Args({2700, 2700, 0})
    ->Args({2800, 2800, 0})
    ->Args({2900, 2900, 0})
    ->Args({3000, 3000, 0})
    ->Args({3100, 3100, 0})
    ->Args({3200, 3200, 0})
    ->Args({3300, 3300, 0})
    ->Args({3400, 3400, 0})
    ->Args({3500, 3500, 0})
    ->Args({3600, 3600, 0})
    ->Args({3700, 3700, 0})
    ->Args({3800, 3800, 0})
    ->Args({3900, 3900, 0})
    ->Args({4000, 4000, 0})
    ->Args({4100, 4100, 0})
    ->Args({4200, 4200, 0})
    ->Args({4300, 4300, 0})
    ->Args({4400, 4400, 0})
    ->Args({4500, 4500, 0})
    ->Args({4600, 4600, 0})
    ->Args({4700, 4700, 0})
    ->Args({4800, 4800, 0})
    ->Args({4900, 4900, 0})
    ->Args({5000, 5000, 0})

    ->Args({100, 100, 1})
    ->Args({200, 200, 1})
    ->Args({300, 300, 1})
    ->Args({400, 400, 1})
    ->Args({500, 500, 1})
    ->Args({600, 600, 1})
    ->Args({700, 700, 1})
    ->Args({800, 800, 1})
    ->Args({900, 900, 1})
    ->Args({1000, 1000, 1})
    ->Args({1100, 1100, 1})
    ->Args({1200, 1200, 1})
    ->Args({1300, 1300, 1})
    ->Args({1400, 1400, 1})
    ->Args({1500, 1500, 1})
    ->Args({1600, 1600, 1})
    ->Args({1700, 1700, 1})
    ->Args({1800, 1800, 1})
    ->Args({1900, 1900, 1})
    ->Args({2000, 2000, 1})
    ->Args({2100, 2100, 1})
    ->Args({2200, 2200, 1})
    ->Args({2300, 2300, 1})
    ->Args({2400, 2400, 1})
    ->Args({2500, 2500, 1})
    ->Args({2600, 2600, 1})
    ->Args({2700, 2700, 1})
    ->Args({2800, 2800, 1})
    ->Args({2900, 2900, 1})
    ->Args({3000, 3000, 1})
    ->Args({3100, 3100, 1})
    ->Args({3200, 3200, 1})
    ->Args({3300, 3300, 1})
    ->Args({3400, 3400, 1})
    ->Args({3500, 3500, 1})
    ->Args({3600, 3600, 1})
    ->Args({3700, 3700, 1})
    ->Args({3800, 3800, 1})
    ->Args({3900, 3900, 1})
    ->Args({4000, 4000, 1})
    ->Args({4100, 4100, 1})
    ->Args({4200, 4200, 1})
    ->Args({4300, 4300, 1})
    ->Args({4400, 4400, 1})
    ->Args({4500, 4500, 1})
    ->Args({4600, 4600, 1})
    ->Args({4700, 4700, 1})
    ->Args({4800, 4800, 1})
    ->Args({4900, 4900, 1})
    ->Args({5000, 5000, 1})

    ->Args({100, 100, 2})
    ->Args({200, 200, 2})
    ->Args({300, 300, 2})
    ->Args({400, 400, 2})
    ->Args({500, 500, 2})
    ->Args({600, 600, 2})
    ->Args({700, 700, 2})
    ->Args({800, 800, 2})
    ->Args({900, 900, 2})
    ->Args({1000, 1000, 2})
    ->Args({1100, 1100, 2})
    ->Args({1200, 1200, 2})
    ->Args({1300, 1300, 2})
    ->Args({1400, 1400, 2})
    ->Args({1500, 1500, 2})
    ->Args({1600, 1600, 2})
    ->Args({1700, 1700, 2})
    ->Args({1800, 1800, 2})
    ->Args({1900, 1900, 2})
    ->Args({2000, 2000, 2})
    ->Args({2100, 2100, 2})
    ->Args({2200, 2200, 2})
    ->Args({2300, 2300, 2})
    ->Args({2400, 2400, 2})
    ->Args({2500, 2500, 2})
    ->Args({2600, 2600, 2})
    ->Args({2700, 2700, 2})
    ->Args({2800, 2800, 2})
    ->Args({2900, 2900, 2})
    ->Args({3000, 3000, 2})
    ->Args({3100, 3100, 2})
    ->Args({3200, 3200, 2})
    ->Args({3300, 3300, 2})
    ->Args({3400, 3400, 2})
    ->Args({3500, 3500, 2})
    ->Args({3600, 3600, 2})
    ->Args({3700, 3700, 2})
    ->Args({3800, 3800, 2})
    ->Args({3900, 3900, 2})
    ->Args({4000, 4000, 2})
    ->Args({4100, 4100, 2})
    ->Args({4200, 4200, 2})
    ->Args({4300, 4300, 2})
    ->Args({4400, 4400, 2})
    ->Args({4500, 4500, 2})
    ->Args({4600, 4600, 2})
    ->Args({4700, 4700, 2})
    ->Args({4800, 4800, 2})
    ->Args({4900, 4900, 2})
    ->Args({5000, 5000, 2})

    ->Args({100, 100, 4})
    ->Args({200, 200, 4})
    ->Args({300, 300, 4})
    ->Args({400, 400, 4})
    ->Args({500, 500, 4})
    ->Args({600, 600, 3})
    ->Args({700, 700, 4})
    ->Args({800, 800, 4})
    ->Args({900, 900, 4})
    ->Args({1000, 1000, 4})
    ->Args({1100, 1100, 4})
    ->Args({1200, 1200, 4})
    ->Args({1300, 1300, 4})
    ->Args({1400, 1400, 4})
    ->Args({1500, 1500, 4})
    ->Args({1600, 1600, 4})
    ->Args({1700, 1700, 4})
    ->Args({1800, 1800, 4})
    ->Args({1900, 1900, 4})
    ->Args({2000, 2000, 4})
    ->Args({2100, 2100, 4})
    ->Args({2200, 2200, 4})
    ->Args({2300, 2300, 4})
    ->Args({2400, 2400, 4})
    ->Args({2500, 2500, 4})
    ->Args({2600, 2600, 4})
    ->Args({2700, 2700, 4})
    ->Args({2800, 2800, 4})
    ->Args({2900, 2900, 4})
    ->Args({3000, 3000, 4})
    ->Args({3100, 3100, 4})
    ->Args({3200, 3200, 4})
    ->Args({3300, 3300, 4})
    ->Args({3400, 3400, 4})
    ->Args({3500, 3500, 4})
    ->Args({3600, 3600, 4})
    ->Args({3700, 3700, 4})
    ->Args({3800, 3800, 4})
    ->Args({3900, 3900, 4})
    ->Args({4000, 4000, 4})
    ->Args({4100, 4100, 4})
    ->Args({4200, 4200, 4})
    ->Args({4300, 4300, 4})
    ->Args({4400, 4400, 4})
    ->Args({4500, 4500, 4})
    ->Args({4600, 4600, 4})
    ->Args({4700, 4700, 4})
    ->Args({4800, 4800, 4})
    ->Args({4900, 4900, 4})
    ->Args({5000, 5000, 4})
    ->Threads(1);
}

BENCHMARK_MAIN();
