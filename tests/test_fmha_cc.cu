#include <gtest/gtest.h>
#include <cobraml2/kernels/fmha_cc.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <test_common/mha.cuh>
#include <curand.h>


using namespace cobraml;
using namespace cute;

TEST(MHA_TEST, kernel) {

    constexpr int head_count{16};
    constexpr int head_dim{64};
    constexpr int B_r{64};
    constexpr int B_c{64};

    int batch_size{56};
    int sequence_length{128};

    using MHAType = kernels::FMHA<head_count, head_dim, B_r, B_c, float>;

    thrust::device_vector<float> q_device{
        test_helpers::create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            test_helpers::seeded_fill_random_uniform<float>(0)
        )
    };

    // thrust::device_vector<float> k_device{
    //     create_projection<float>(
    //         head_count, head_dim, batch_size, sequence_length,
    //         seeded_fill_fn(1)
    //     )
    // };

    // thrust::device_vector<float> v_device{
    //     create_projection<float>(
    //         head_count, head_dim, batch_size, sequence_length,
    //         seeded_fill_fn(2)
    //     )
    // };

    // auto fill_zero_ptr{fill_zero<float>};

    // thrust::device_vector<float> o_device{
    //     create_projection<float>(
    //         head_count, head_dim, batch_size, sequence_length,
    //         fill_zero_ptr
    //     )
    // };

    // MHAType mha{};

    // mha(
    //     thrust::raw_pointer_cast(q_device.data()), 
    //     thrust::raw_pointer_cast(k_device.data()), 
    //     thrust::raw_pointer_cast(v_device.data()), 
    //     thrust::raw_pointer_cast(o_device.data()), 
    //     batch_size, 
    //     sequence_length
    // );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}