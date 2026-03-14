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
};
