//
// Created by sriram on 7/6/25.
//

#ifndef COPY_BUILDER_H
#define COPY_BUILDER_H

#include <cute/layout.hpp>
#include <cute/tensor.hpp>


namespace cobraml {
    using namespace cute;

    template<
        typename LoadDataType,
        typename DataType,
        size_t thread_count_x,
        size_t thread_count_y,
        bool k_major_a = false,
        bool k_major_b = false
    >
    struct AsyncTiledCopyBuilder {
        // TODO make more descriptive
        static_assert(sizeof(LoadDataType) % sizeof(DataType) == 0, "DataType must be a factor of LoadDataType");
        static constexpr size_t elements_per_load{sizeof(LoadDataType) / sizeof(DataType)};

        template<bool is_k_major>
        using ThreadStrideType = std::conditional_t<
            is_k_major,
            Stride<Int<thread_count_y>, _1>,
            Stride<_1, Int<thread_count_x> >
        >;

        template<bool is_k_major>
        using ThreadLayoutType = Layout<
            Shape<Int<thread_count_x>, Int<thread_count_y> >,
            ThreadStrideType<is_k_major>
        >;

        template<bool is_k_major>
        using ValueLayoutType = std::conditional_t<
            is_k_major,
            Layout<Shape<_1, Int<elements_per_load> >>,
            Layout<Shape<Int<elements_per_load>, _1>>
        >;

        template<bool is_k_major>
        static auto get_tiled_copy() {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<LoadDataType>, DataType>{},
                ThreadLayoutType<is_k_major>{},
                ValueLayoutType<is_k_major>{}
            );
        }

        static auto get_tiled_copy_A() {
            return get_tiled_copy<k_major_a>();
        }

        static auto get_tiled_copy_B() {
            return get_tiled_copy<k_major_b>();
        }
    };
}

#endif //COPY_BUILDER_H
