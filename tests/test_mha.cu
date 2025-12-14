#include <gtest/gtest.h>
#include <cobraml2/kernels/mha_naive.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>


using namespace cobraml::kernels;
using namespace cute;

template<typename DType>
void fill_zero(float * data, int length){
    cudaMemset(data, 0, length * sizeof(DType));    
    cudaDeviceSynchronize(); 
}

void fill_projection_random_uniform(float * data, int length, int seed){
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, data, length);
    curandDestroyGenerator(gen);
    cudaDeviceSynchronize(); 
}

template<typename DType>
thrust::device_vector<DType> create_projection(
    int head_count, 
    int head_dim, 
    int batch_size, 
    int sequence_length,
    auto fill_fn
){
    int total_length{head_count * head_dim * batch_size * sequence_length};
    thrust::device_vector<DType> device_vec(total_length);
    fill_fn(thrust::raw_pointer_cast(device_vec.data()), total_length);

    return device_vec;
}

auto seeded_fill_fn(int seed) {
    return [=](float * data, int length){
        fill_projection_random_uniform(data, length, seed);
    };
}

TEST(MHA_TEST, kernel) {

    constexpr int head_count{16};
    constexpr int head_dim{64};
    constexpr int B_r{64};
    constexpr int B_c{64};

    int batch_size{56};
    int sequence_length{128};

    using MHAType = MHA<head_count, head_dim, B_r, B_c, float>;

    thrust::device_vector<float> q_device{
        create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            seeded_fill_fn(0)
        )
    };

    thrust::device_vector<float> k_device{
        create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            seeded_fill_fn(1)
        )
    };

    thrust::device_vector<float> v_device{
        create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            seeded_fill_fn(2)
        )
    };

    auto fill_zero_ptr{fill_zero<float>};

    thrust::device_vector<float> o_device{
        create_projection<float>(
            head_count, head_dim, batch_size, sequence_length,
            fill_zero_ptr
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