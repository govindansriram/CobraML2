//
// Created by sriram on 12/19/24.
//

#ifndef ENUMS_H
#define ENUMS_H
#include <cstdint>
#include <stdexcept>

namespace cobraml::core {
    enum Device {
        CPU,    // standard naive implementations
        GPU,    // GPU implementations
        CPU_X   // Accelerated CPU implementation
    };

    enum Dtype {
        INT8,
        INT16,
        INT32,
        INT64,
        FLOAT32,
        FLOAT64,
        INVALID
    };

    bool operator<(Dtype lhs, Dtype rhs);

    constexpr unsigned char dtype_to_bytes(Dtype const type) {
        switch (type) {
            case INT8: return 1;
            case INT16: return 2;
            case INT32: return 4;
            case INT64: return 8;
            case FLOAT32: return 4;
            case FLOAT64: return 8;
            case INVALID: return 0;
        }

        return 0;
    }

    extern unsigned char func_pos;

    std::string dtype_to_string(Dtype dtype);
    std::string device_to_string(Device device);

    inline void is_invalid(Dtype const dtype) {
        if (dtype == INVALID)
            throw std::runtime_error("invalid dtype provided");
    }

    template<typename T>
    struct get_dtype_from_type {
        static constexpr Dtype type = INVALID;
    };

    template<>
    struct get_dtype_from_type<int8_t> {
        static constexpr Dtype type = INT8;
    };

    template<>
    struct get_dtype_from_type<int16_t> {
        static constexpr Dtype type = INT16;
    };

    template<>
    struct get_dtype_from_type<int32_t> {
        static constexpr Dtype type = INT32;
    };

    template<>
    struct get_dtype_from_type<int64_t> {
        static constexpr Dtype type = INT64;
    };

    template<>
    struct get_dtype_from_type<float> {
        static constexpr Dtype type = FLOAT32;
    };

    template<>
    struct get_dtype_from_type<double> {
        static constexpr Dtype type = FLOAT64;
    };

#define INSTANTIATE_OPERATOR(operation)\
    template Brarray operation<float>(float other) const;\
    template Brarray operation<double>(double other) const;\
    template Brarray operation<int64_t>(int64_t other) const;\
    template Brarray operation<int32_t>(int32_t other) const;\
    template Brarray operation<int16_t>(int16_t other) const;\
    template Brarray operation<int8_t>(int8_t other) const;\


#define INSTANTIATE_INPLACE_OPERATOR(operation)\
    template void operation<float>(Brarray &input, const float other);\
    template void operation<double>(Brarray &input, const double other);\
    template void operation<int64_t>(Brarray &input, const int64_t other);\
    template void operation<int32_t>(Brarray &input, const int32_t other);\
    template void operation<int16_t>(Brarray &input, const int16_t other);\
    template void operation<int8_t>(Brarray &input, const int8_t other);\

}

#endif //ENUMS_H
