#include <gtest/gtest.h>
#include "cobraml2/kernels/mha_naive.cuh"
#include <cute/util/print.hpp>

using namespace cobraml::kernels;
using namespace cute;

TEST(MHA_TEST, kernel) {

    using MHAType = MHA<8, 64, 64, 64, float>;

    MHAType mha{};

    mha(nullptr, nullptr, nullptr, nullptr, 30, 100);

    cudaDeviceSynchronize();
}