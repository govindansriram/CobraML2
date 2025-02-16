//
// Created by Sriram Govindan on 12/23/24.
//

#include <gtest/gtest.h>

#include "enums.h"

TEST(DTYPE, validity) {
    ASSERT_THROW(cobraml::core::INVALID < cobraml::core::INT8, std::runtime_error);
}

TEST(DTYPE, is_invalid) {
    ASSERT_THROW(cobraml::core::is_invalid(cobraml::core::INVALID), std::runtime_error);
    ASSERT_NO_THROW(cobraml::core::is_invalid(cobraml::core::FLOAT32));
}

TEST(DTYPE, comp_operators) {
    ASSERT_GE(cobraml::core::INT16, cobraml::core::INT8);
    ASSERT_GE(cobraml::core::INT32, cobraml::core::INT16);
    ASSERT_GE(cobraml::core::INT64, cobraml::core::INT32);
    ASSERT_GE(cobraml::core::FLOAT32, cobraml::core::INT64);
    ASSERT_GE(cobraml::core::FLOAT64, cobraml::core::FLOAT32);
}
