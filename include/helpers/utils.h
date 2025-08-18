//
// Created by sriram on 8/11/25.
//

#ifndef COBRAML_DIM_H
#define COBRAML_DIM_H

#include <array>


namespace cobraml {

    template<int value = INT_MIN>
    struct Dim {
        static constexpr bool is_comptime = value != INT_MIN;
        static constexpr int comp_dim = value;
        int run_dim;

        Dim(): run_dim(value){}
        explicit Dim(const int runtime_value): run_dim(runtime_value) {}
    };

    template<int m, int n, int k>
    struct Shape {

        using DimM = Dim<m>;
        using DimN = Dim<n>;
        using DimK = Dim<k>;

        const DimM dim_m;
        const DimN dim_n;
        const DimK dim_k;
        int runtime_shape[3];

        static constexpr bool a_is_static{DimM::is_comptime && DimK::is_comptime};
        static constexpr bool b_is_static{DimN::is_comptime && DimK::is_comptime};
        static constexpr bool c_is_static{DimM::is_comptime && DimN::is_comptime};
        static constexpr bool is_comptime{b_is_static && c_is_static};

        static constexpr int comp_shape[3]{m, n, k};

        Shape(const DimM dim_m, const DimN dim_n, const DimK dim_k):
            dim_m(dim_m), dim_n(dim_n), dim_k(dim_k), runtime_shape{dim_m.run_dim, dim_n.run_dim, dim_k.run_dim} {}

        Shape():
            dim_m{}, dim_n{}, dim_k{}, runtime_shape{dim_m.run_dim, dim_n.run_dim, dim_k.run_dim} {}
    };

    template<bool a_k_major, bool b_k_major, bool c_n_major>
    struct MatrixMajor {
        /**
         * All three of these tell me if the matrix is row major or
         * not
         */
        static constexpr bool a_is_k_major{!a_k_major};
        static constexpr bool b_is_k_major{!b_k_major};
        static constexpr bool c_is_n_major{!c_n_major};
    };

    template<bool c_n_major>
    using TN = MatrixMajor<true, true, c_n_major>;

    template<bool c_n_major>
    using TT = MatrixMajor<true, false, c_n_major>;

    template<bool c_n_major>
    using NT = MatrixMajor<false, false, c_n_major>;

    template<bool c_n_major>
    using NN = MatrixMajor<false, true, c_n_major>;

}

#endif //COBRAML_DIM_H