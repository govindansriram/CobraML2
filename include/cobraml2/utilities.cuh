#pragma once
#include <cute/layout.hpp>
#include <cute/tensor.hpp>


namespace cobraml {

    using namespace cute;

    bool is_row_major(std::pair<size_t, size_t> stride, size_t fast_mode) {
        if (stride.first != fast_mode)
            return false;
        if (stride.second != 1)
            return false;
        return true;
    }

    template<typename StrideType>
    bool equivalent_stride(
        std::pair<size_t, size_t> stride, 
        StrideType row_major_stride){
        return is_row_major(stride, get<0>(row_major_stride));
    }

    template<
        typename AType, 
        typename BType, 
        typename CType,
        size_t static_N,
        size_t static_K
    >
    struct GEMMShapeManager{

        using NType = Int<static_N>;
        using KType = Int<static_K>;

        using ATensorType = AType;
        using BTensorType = BType;
        using CTensorType = CType;

        size_t M;

        GEMMShapeManager(size_t M): M(M){}

        /**
         * @brief Initialize the The A, B, and C tensors
         * with row major layout
         * 
         * @param a ptr in gmem to A
         * @param b ptr in gmem to B
         * @param c ptr in gmem to C
         * @return auto the three tensors
         */
        auto init_tensors(
            const AType * __restrict__ a,
            const BType *__restrict__ b,
            CType *__restrict__ c
        ){

            auto layout_a{
                make_layout(make_shape(M, KType{}), LayoutRight{})
            };
            auto layout_b{
                make_layout(make_shape(NType{}, KType{}), LayoutRight{})
            };
            auto layout_c{
                make_layout(make_shape(M, NType{}), LayoutRight{})
            };

            auto a_tensor{
                make_tensor(make_gmem_ptr<AType>(a), layout_a)
            };
            auto b_tensor{
                make_tensor(make_gmem_ptr<BType>(b), layout_b)
            };
            auto c_tensor{
                make_tensor(make_gmem_ptr<CType>(c), layout_c)
            };

            return cute::make_tuple(a_tensor, b_tensor, c_tensor);
        }
    };
};
