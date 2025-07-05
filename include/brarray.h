//
// Created by sriram on 1/25/25.
//

#ifndef BARRAY_H
#define BARRAY_H

#include <vector>
#include <cute/layout.hpp>
#include <thrust/device_vector.h>
#include <utility>

namespace cobraml {
    using namespace cute;

    template<
        typename Shape,
        typename Stride,
        size_t... Modes>
    static constexpr size_t calculate_memory_span_impl(
        const Layout<Shape, Stride> &layout,
        std::index_sequence<Modes...>) {
        return (((shape<Modes>(layout) - 1) * stride<Modes>(layout)) + ... + 1);
    }

    template<
        typename Shape,
        typename Stride>
    static constexpr size_t calculate_memory_span(Layout<Shape, Stride> layout) {
        auto flattened_layout = flatten(layout);
        constexpr size_t flattend_rank = rank(flattened_layout);
        return calculate_memory_span_impl(flattened_layout, std::make_index_sequence<flattend_rank>{});
    }

    template<
        typename ArrayType,
        typename LayoutShape,
        typename LayoutStride
    >
    class Brarray {
        Layout<LayoutShape, LayoutStride> layout_;
        thrust::device_vector<ArrayType> buffer_{};

        ArrayType *get_ptr() {
            return buffer_.data().get();
        }

    public:
        /**
         * This constructor takes complete ownership of a buffer, leaving the original buffer
         * invalid
         * @param layout The CuTe Layout of the buffer
         * @param buffer The underlying buffer
         */
        Brarray(
            const Layout<LayoutShape, LayoutStride> &layout,
            thrust::device_vector<ArrayType> &buffer
        ): layout_(layout), buffer_(std::move(buffer)) {
            // rounds to nearest alignment value
            const size_t memory_span{calculate_memory_span(layout_)};

            if (memory_span > buffer_.size()) {
                std::stringstream stream;

                stream << "buffer does not have enough data to support layout " << layout
                << " minimum required bytes are " << memory_span * sizeof(ArrayType)
                << ", " << buffer_.size() * sizeof(ArrayType) << " was provided";

                throw std::runtime_error(stream.str());
            }
        }

        /**
         * Allocate based on provided Layout with data provided from vector
         * @param layout The CuTe Layout of the buffer
         * @param vector The data that will be copied to device
         */
        Brarray(
            const Layout<LayoutShape, LayoutStride> &layout,
            const std::vector<ArrayType> &vector): layout_(layout), buffer_(thrust::device_vector<ArrayType>(vector)) {
        }
    };
}

#endif //BARRAY_H
