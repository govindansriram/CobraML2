//
// Created by sriram on 12/19/24.
//


#include "matrix.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include "allocator.h"
#include "enums.h"
#include "math_dis.h"

namespace cobraml::core {
    struct Matrix::MatrixImpl {
        size_t offset = 0;
        size_t rows = 0;
        size_t columns = 0;
        Device device = CPU;
        Dtype dtype = INVALID;
        std::shared_ptr<Buffer> buffer = nullptr;
        Math * m_dispatcher = nullptr;

        MatrixImpl(size_t const rows, size_t const columns, Device const device, Dtype const type): rows(rows),
            columns(columns),
            device(device),
            dtype(type),
            buffer(std::make_shared<Buffer>(rows * columns * dtype_to_bytes(type), device)),
            m_dispatcher(get_math_kernels(device)) {
        }

        MatrixImpl() = default;
        MatrixImpl(const MatrixImpl&) = default;
        MatrixImpl& operator=(const MatrixImpl&) = default;

        [[nodiscard]] void *get_raw_buffer() const {
            return static_cast<char *>(buffer->get_p_buffer()) + offset;
        }
    };

    Matrix::Matrix(size_t rows, size_t columns, Device device, Dtype dtype): impl(
        std::make_unique<MatrixImpl>(rows, columns, device, dtype)) {
        is_invalid(dtype);
    }

    Dtype Matrix::get_dtype() const {
        return this->impl->dtype;
    }

    Device Matrix::get_device() const {
        return this->impl->device;
    }

    Matrix::Shape Matrix::get_shape() const {
        Shape sh{};
        sh.columns = impl->columns;
        sh.rows = impl->rows;
        return sh;
    }


    Matrix::~Matrix() = default;

    void *Matrix::get_raw_buffer() const {
        return impl->get_raw_buffer();
    }

    Matrix::Matrix(): impl(std::make_unique<MatrixImpl>()) {}

    void print_num(void *buffer, Dtype const dtype) {
        switch (dtype) {
            case INT8: {
                const auto num = *static_cast<char *>(buffer);
                std::cout << static_cast<int>(num);
                return;
            }
            case INT16: {
                std::cout << *static_cast<int16_t *>(buffer);
                return;
            }
            case INT32: {
                std::cout << *static_cast<int32_t *>(buffer);
                return;
            }
            case INT64: {
                std::cout << *static_cast<int64_t *>(buffer);
                return;
            }
            case FLOAT32: {
                std::cout << *static_cast<float *>(buffer);
                return;
            }
            case FLOAT64: {
                std::cout << *static_cast<double *>(buffer);
                return;
            }
            case INVALID: {
                is_invalid(dtype);
            }
        }
    }

    Matrix::Matrix(Matrix const &other):impl(std::make_unique<MatrixImpl>()) {
        impl->columns = other.impl->columns;
        impl->rows = other.impl->rows;
        impl->dtype = other.impl->dtype;
        impl->device = other.impl->device;
        impl->buffer = other.impl->buffer;
        impl->m_dispatcher = other.impl->m_dispatcher;
    }

    Matrix& Matrix::operator=(const Matrix &other) {
        if (this != &other) {
            impl->columns = other.impl->columns;
            impl->rows = other.impl->rows;
            impl->dtype = other.impl->dtype;
            impl->device = other.impl->device;
            impl->buffer = other.impl->buffer;
            impl->m_dispatcher = other.impl->m_dispatcher;
        }

        return *this;
    }

    void Matrix::replace_segment(const void *source, size_t const offset, size_t const bytes) const {
        this->impl->buffer->overwrite(source, bytes, offset);
    }


    void gemv(const Matrix &matrix, const Matrix &vector, Matrix &result, const void * alpha, const void * beta) {
        if (vector.impl->rows != 1) {
            throw std::runtime_error("vector is a matrix");
        }

        if (result.impl->rows != 1) {
            throw std::runtime_error("result is a matrix");
        }

        if (matrix.impl->columns != vector.impl->columns) {
            throw std::runtime_error("vector and matrix have different columns lengths");
        }

        if (matrix.impl->rows != result.impl->columns) {
            throw std::runtime_error("result must be size 1, rows(matrix)");
        }

        if (matrix.impl->device != vector.impl->device || matrix.impl->device != result.impl->device) {
            throw std::runtime_error("vector, matrix and result are not on the same device");
        }

        if (matrix.impl->dtype != vector.impl->dtype || matrix.impl->dtype != result.impl->dtype) {
            throw std::runtime_error("vector, matrix and result share different dtypes");
        }

        result.impl->m_dispatcher->gemv(
            matrix.get_raw_buffer(),
            vector.get_raw_buffer(),
            result.get_raw_buffer(),
            alpha,
            beta,
            matrix.impl->rows,
            matrix.impl->columns,
            matrix.impl->dtype);
    }

    void print_details(Device const device, Dtype const dtype, size_t const rows, size_t const columns) {
        std::cout << "############## Details ##############\n";
        std::cout << "Shape: " << "(" << rows << ", " << columns << ")" << '\n';
        std::cout << "Device: " << device_to_string(device) << '\n';
        std::cout << "Dtype: " << dtype_to_string(dtype) << '\n';
        std::cout << "#####################################\n";
    }

    void Matrix::print(bool const hide_middle) const {
        print_details(impl->device, impl->dtype, impl->rows, impl->columns);
        unsigned char const shift = dtype_to_bytes(impl->dtype);
        auto dec3 = [this](size_t const x, size_t &start) {
            if (x == 1)
                start = impl->rows - 3;
        };

        std::cout << "[\n";
        for (size_t x = 0; x < 2; ++x) {
            size_t start = 0;
            size_t end = impl->rows;

            if (impl->rows > 20 && hide_middle) {
                dec3(x, start);
                end = start + 3;
            }

            for (; start < end; ++start) {
                std::cout << "   [";

                for (size_t y = 0; y < 2; ++y) {
                    size_t start_inner = 0;
                    size_t end_inner = impl->columns;

                    bool inner_hiding{false};

                    if (impl->columns > 20 && hide_middle) {
                        dec3(y, start_inner);
                        end_inner = start_inner + 3;
                        inner_hiding = true;
                    }

                    for (; start_inner < end_inner; ++start_inner) {
                        std::cout << std::fixed << std::setprecision(5);

                        auto buff = static_cast<char *>(impl->get_raw_buffer());
                        buff += (start * impl->columns + start_inner) * shift;
                        print_num(buff, impl->dtype);

                        if (start_inner != end_inner - 1 || (y == 0 && inner_hiding)) {
                            std::cout << ", ";
                        }
                    }

                    if (impl->columns <= 20 || !hide_middle)
                        break;

                    if (y == 0) {
                        std::cout << "..., ";
                    }
                }

                std::cout << "]";
                std::cout << "\n";
            }

            if (impl->rows <= 20 || !hide_middle)
                break;

            if (x == 0) {
                std::cout << "   ..." << "\n";
            }
        }

        std::cout << "]\n\n";
    }

    static size_t sum_vec(std::vector<size_t> const &shape) {

        if (shape.empty())
            return 0;

        size_t sum{1};
        for (auto const &num: shape)
            sum *= num;

        return sum;
    }

    struct Tensor::TensorImpl {
        size_t offset = 0;
        std::vector<size_t> shape;
        Device device = CPU;
        Dtype dtype = INVALID;
        std::shared_ptr<Buffer> buffer = nullptr;
        Math * m_dispatcher = nullptr;

        TensorImpl() = default;
        TensorImpl(const TensorImpl&) = default;
        TensorImpl& operator=(const TensorImpl&) = default;

        TensorImpl(std::vector<size_t> shape, Device const device, Dtype const dtype):
            shape(std::move(shape)), device(device), dtype(dtype) {
            size_t const sum = sum_vec(shape);
            buffer = std::make_shared<Buffer>(sum * dtype_to_bytes(dtype), device);
            m_dispatcher = get_math_kernels(device);
        }
    };

    Tensor::Tensor(std::vector<size_t> shape, Device device, Dtype dtype):
        impl(std::make_unique<TensorImpl>(std::move(shape), device, dtype)){
    }

    void Tensor::test_nothing() {
        auto const m = Matrix();
        // Matrix const m(1, 1, this->impl->device, this->impl->dtype);
        std::cout << (m.impl->buffer == nullptr) << std::endl;
    }

    Tensor::~Tensor() = default;
}
