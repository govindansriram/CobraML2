//
// Created by sriram on 1/26/25.
//
#include <gtest/gtest.h>
#include "test.h"
#include "brarray.h"

TEST(ArrayTestFunctionals, test_dtype) {

    using namespace cobraml;

    constexpr Layout<Shape<_32, _4>, Stride<_4, _1>> layout{};
    constexpr float value = 0.f;

    Brarray br(layout, value);
}


