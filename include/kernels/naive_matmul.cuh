//
// Created by sriram on 8/17/25.
//

#ifndef COBRAML_NAIVE_MATMUL_H
#define COBRAML_NAIVE_MATMUL_H

namespace cobraml {
    template<
        typename DataType,
        typename AccumulatorType
    >
    __global__ void naive_matmul_TTT(
        int const m,
        int const n,
        int const k,
        DataType *a,
        DataType *b,
        AccumulatorType *c
    ) {
        for (int i{0}; i < m; ++i) {
            for (int j{0}; j < n; ++j) {
                DataType sm{0};
                for (int kk{0}; kk < k; ++kk) {
                    sm += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = static_cast<AccumulatorType>(sm);
            }
        }
    }

    template<
        typename DataType,
        typename AccumulatorType
    >
    __global__ void naive_matmul_TNN(
        int const m,
        int const n,
        int const k,
        DataType *a,
        DataType *b,
        AccumulatorType *c
    ) {
        for (int i{0}; i < m; ++i) {
            for (int j{0}; j < n; ++j) {
                DataType sm{0};
                for (int kk{0}; kk < k; ++kk) {
                    sm += a[i * k + kk] * b[j * k + kk];
                }
                c[i + j * m] = static_cast<AccumulatorType>(sm);
            }
        }
    }

    template<
        typename DataType,
        typename AccumulatorType
    >
    __global__ void naive_matmul_TNT(
        int const m,
        int const n,
        int const k,
        DataType *a,
        DataType *b,
        AccumulatorType *c
    ) {
        for (int i{0}; i < m; ++i) {
            for (int j{0}; j < n; ++j) {
                DataType sm{0};
                for (int kk{0}; kk < k; ++kk) {
                    sm += a[i * k + kk] * b[j * k + kk];
                }
                c[i * n + j] = static_cast<AccumulatorType>(sm);
            }
        }
    }
}

#endif //COBRAML_NAIVE_MATMUL_H
