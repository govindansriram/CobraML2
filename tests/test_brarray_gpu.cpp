//
// Created by sriram on 4/21/25.
//
#include <gtest/gtest.h>
#include "brarray.h"

TEST(ArrayTestFunctionals, test_device) {
    cobraml::core::Brarray arr(cobraml::core::Device::CPU, cobraml::core::Dtype::INT8, {10});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CPU);

    arr = cobraml::core::Brarray(cobraml::core::Device::CUDA, cobraml::core::Dtype::INT8, {10});
    ASSERT_EQ(arr.get_device(), cobraml::core::Device::CUDA);
}
