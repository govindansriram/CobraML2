//
// Created by sriram on 4/26/25.
//

#ifndef INSTANTIATE_H
#define INSTANTIATE_H

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


#define INSTANTIATE_GET_ITEM()                                         \
    template float Brarray::item<float>() const;                       \
    template double Brarray::item<double>() const;                     \
    template int8_t Brarray::item<int8_t>() const;                     \
    template int16_t Brarray::item<int16_t>() const;                   \
    template int32_t Brarray::item<int32_t>() const;                   \
    template int64_t Brarray::item<int64_t>() const;                   \


#define INSTANTIATE_GEMV_WHOLE()                                                                                                \
    template void gemv<float>(Brarray &result, const Brarray &matrix, const Brarray &vector, float alpha, float beta);          \
    template void gemv<double>(Brarray &result, const Brarray &matrix, const Brarray &vector, double alpha, double beta);       \
    template void gemv<int8_t>(Brarray &result, const Brarray &matrix, const Brarray &vector, int8_t alpha, int8_t beta);       \
    template void gemv<int16_t>(Brarray &result, const Brarray &matrix, const Brarray &vector, int16_t alpha, int16_t beta);    \
    template void gemv<int32_t>(Brarray &result, const Brarray &matrix, const Brarray &vector, int32_t alpha, int32_t beta);    \
    template void gemv<int64_t>(Brarray &result, const Brarray &matrix, const Brarray &vector, int64_t alpha, int64_t beta);    \


#define INSTANTIATE_GEMV_PARTIAL()                                                                                \
    template Brarray gemv<float>(const Brarray &matrix, const Brarray &vector, float alpha, float beta);          \
    template Brarray gemv<double>(const Brarray &matrix, const Brarray &vector, double alpha, double beta);       \
    template Brarray gemv<int8_t>(const Brarray &matrix, const Brarray &vector, int8_t alpha, int8_t beta);       \
    template Brarray gemv<int16_t>(const Brarray &matrix, const Brarray &vector, int16_t alpha, int16_t beta);    \
    template Brarray gemv<int32_t>(const Brarray &matrix, const Brarray &vector, int32_t alpha, int32_t beta);    \
    template Brarray gemv<int64_t>(const Brarray &matrix, const Brarray &vector, int64_t alpha, int64_t beta);    \

#define INSTANTIATE_GEMM_WHOLE()                                                                                                      \
    template void gemm<float>(Brarray &result, const Brarray &matrix, const Brarray &other_matrix, float alpha, float beta);          \
    template void gemm<double>(Brarray &result, const Brarray &matrix, const Brarray &other_matrix, double alpha, double beta);       \
    template void gemm<int8_t>(Brarray &result, const Brarray &matrix, const Brarray &other_matrix, int8_t alpha, int8_t beta);       \
    template void gemm<int16_t>(Brarray &result, const Brarray &matrix, const Brarray &other_matrix, int16_t alpha, int16_t beta);    \
    template void gemm<int32_t>(Brarray &result, const Brarray &matrix, const Brarray &other_matrix, int32_t alpha, int32_t beta);    \
    template void gemm<int64_t>(Brarray &result, const Brarray &matrix, const Brarray &other_matrix, int64_t alpha, int64_t beta);    \


#define INSTANTIATE_GEMM_PARTIAL()                                                                                   \
    template Brarray gemm<float>(const Brarray &matrix, const Brarray &other_matrix, float alpha, float beta);          \
    template Brarray gemm<double>(const Brarray &matrix, const Brarray &other_matrix, double alpha, double beta);       \
    template Brarray gemm<int8_t>(const Brarray &matrix, const Brarray &other_matrix, int8_t alpha, int8_t beta);       \
    template Brarray gemm<int16_t>(const Brarray &matrix, const Brarray &other_matrix, int16_t alpha, int16_t beta);    \
    template Brarray gemm<int32_t>(const Brarray &matrix, const Brarray &other_matrix, int32_t alpha, int32_t beta);    \
    template Brarray gemm<int64_t>(const Brarray &matrix, const Brarray &other_matrix, int64_t alpha, int64_t beta);    \

#endif //INSTANTIATE_H
