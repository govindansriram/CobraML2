//
// Created by sriram on 12/19/24.
//

#ifndef ENUMS_H
#define ENUMS_H
#include <stdexcept>

namespace cobraml::core {
    enum Device {
        CPU,          // standard naive implementations
        CUDA,         // GPU implementations
        CPU_X,        // Accelerated CPU implementation
        PINNED_CPU,   // CPU with pinned memory still requires cuda
    };

    inline bool device_is_host(const Device device) {
        if (device == CPU) return true;
        if (device == CPU_X) return true;
        if (device == PINNED_CPU) return true;

        return false;
    }

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
    template Brarray operation<float>(float other) const;      \
    template Brarray operation<double>(double other) const;    \
    template Brarray operation<int64_t>(int64_t other) const;  \
    template Brarray operation<int32_t>(int32_t other) const;  \
    template Brarray operation<int16_t>(int16_t other) const;  \
    template Brarray operation<int8_t>(int8_t other) const;    \


#define INSTANTIATE_INPLACE_OPERATOR(operation)\
    template void operation<float>(Brarray &input, const float other);     \
    template void operation<double>(Brarray &input, const double other);   \
    template void operation<int64_t>(Brarray &input, const int64_t other); \
    template void operation<int32_t>(Brarray &input, const int32_t other); \
    template void operation<int16_t>(Brarray &input, const int16_t other); \
    template void operation<int8_t>(Brarray &input, const int8_t other);   \


#define INSTANTIATE_VECTOR_CONSTRUCTOR()                                                                                                    \
    template Brarray::Brarray<float>(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<float> &data);         \
    template Brarray::Brarray<double>(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<double> &data);       \
    template Brarray::Brarray<int64_t>(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<int64_t> &data);     \
    template Brarray::Brarray<int32_t>(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<int32_t> &data);     \
    template Brarray::Brarray<int16_t>(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<int16_t> &data);     \
    template Brarray::Brarray<int8_t>(Device device, Dtype dtype, std::vector<size_t> const &shape, const std::vector<int8_t> &data);       \


#define INSTANTIATE_GET_BUFFER()                                  \
    template float * Brarray::get_buffer<float>() const;          \
    template double * Brarray::get_buffer<double>() const;        \
    template int64_t * Brarray::get_buffer<int64_t>() const;      \
    template int32_t * Brarray::get_buffer<int32_t>() const;      \
    template int16_t * Brarray::get_buffer<int16_t>() const;      \
    template int8_t * Brarray::get_buffer<int8_t>() const;        \


#define INSTANTIATE_SET_ITEM()                                        \
    template void Brarray::set_item<float>(float value);              \
    template void Brarray::set_item<double>(double value);            \
    template void Brarray::set_item<int64_t>(int64_t value);          \
    template void Brarray::set_item<int32_t>(int32_t value);          \
    template void Brarray::set_item<int16_t>(int16_t value);          \
    template void Brarray::set_item<int8_t>(int8_t value);            \




}

#endif //ENUMS_H
