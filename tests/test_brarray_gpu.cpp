//
// Created by sriram on 4/21/25.
//
#include <gtest/gtest.h>
#include "brarray.h"

TEST(CudaArrayTestFunctionals, test_device) {
    const std::vector<float> vec{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    const cobraml::core::Brarray tensor(
            cobraml::core::CUDA,
            cobraml::core::FLOAT32,
            {2, 4, 3}, vec);

    for (size_t i{0}; i < tensor.get_shape()[0]; ++i) {
        auto matrix{tensor[i]};
        for (size_t j{0}; j < matrix.get_shape()[0]; ++j) {
            auto vector{matrix[j]};
            for (size_t k{0}; k < vector.get_shape()[0]; ++k) {
                auto scalar{vector[k]};
                float expected{vec[i * 12 + j * 3 + k]};
                ASSERT_EQ(scalar.item<float>(), expected);
            }
        }
    }

    auto scal{tensor[0][2][2]};
    scal.set_item(10.3f);

    auto res = cobraml::core::gemv(tensor[0], tensor[0][0], 1.0f, 1.0f);


    std::cout << res;
    std::cout << tensor;
}
