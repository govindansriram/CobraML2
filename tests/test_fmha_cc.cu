#include <gtest/gtest.h>
#include <cobraml2/kernels/fmha_cc.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <test_common/mha.cuh>
#include <curand.h>


using namespace cobraml;
using namespace cute;

template<
    int head_count,
    int head_dim,
    int B_r,
    int B_c
>
void test_fmha(int batch_size, int sequence_length){
    using MHAType = kernels::FMHA<head_count, head_dim, B_r, B_c, float>;

    thrust::device_vector<float> q_device{
        test_helpers::create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            test_helpers::seeded_fill_random_uniform<float>(0)
        )
    };

    thrust::device_vector<float> k_device{
        test_helpers::create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            test_helpers::seeded_fill_random_uniform<float>(1)
        )
    };

    thrust::device_vector<float> v_device{
        test_helpers::create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            test_helpers::seeded_fill_random_uniform<float>(2)
        )
    };

    thrust::device_vector<float> o_device{
        test_helpers::create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            test_helpers::fill_zero<float>
        )
    };

    MHAType mha{};

    mha(
        thrust::raw_pointer_cast(q_device.data()), 
        thrust::raw_pointer_cast(k_device.data()), 
        thrust::raw_pointer_cast(v_device.data()), 
        thrust::raw_pointer_cast(o_device.data()), 
        batch_size, 
        sequence_length
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

TEST(FMHA_CC, H16_Hd64_Br64_Bc64_Bs56_Sq128) {
    test_fmha<16, 64, 64, 64>(
        56, 128
    );
}