#include <gtest/gtest.h>
#include <cobraml2/kernels/mha.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>


using namespace cobraml::kernels;

// Fill device memory with zeros
template<typename DType>
void fill_zero(float* data, int length) {
    cudaMemset(data, 0, length * sizeof(DType));
    cudaDeviceSynchronize();
}

// Fill device memory with random uniform values
void fill_random_uniform(float* data, int length, int seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, data, length);
    curandDestroyGenerator(gen);
    cudaDeviceSynchronize();
}

// Create a projection tensor on device with shape [B, H, N, d]
// Layout: batch, heads, sequence_length, head_dim
template<typename DType>
thrust::device_vector<DType> create_projection(
    int batch_size,
    int head_count,
    int sequence_length,
    int head_dim,
    auto fill_fn
) {
    int total_length = batch_size * head_count * sequence_length * head_dim;
    thrust::device_vector<DType> device_vec(total_length);
    fill_fn(thrust::raw_pointer_cast(device_vec.data()), total_length);
    return device_vec;
}

// Returns a fill function with a specific seed for reproducibility
auto seeded_fill_fn(int seed) {
    return [=](float* data, int length) {
        fill_random_uniform(data, length, seed);
    };
}

TEST(MHA_NAIVE_TEST, forward_pass) {
    // Test configuration
    constexpr int head_count = 16;   // H
    constexpr int head_dim = 64;     // d
    int batch_size = 56;             // B
    int sequence_length = 128;       // N

    // Create Q tensor [B, H, N, d]
    thrust::device_vector<float> q_device = create_projection<float>(
        batch_size, head_count, sequence_length, head_dim,
        seeded_fill_fn(0)
    );

    // Create K tensor [B, H, N, d]
    thrust::device_vector<float> k_device = create_projection<float>(
        batch_size, head_count, sequence_length, head_dim,
        seeded_fill_fn(1)
    );

    // Create V tensor [B, H, N, d]
    thrust::device_vector<float> v_device = create_projection<float>(
        batch_size, head_count, sequence_length, head_dim,
        seeded_fill_fn(2)
    );

    // Create O tensor [B, H, N, d] initialized to zero
    auto fill_zero_fn = fill_zero<float>;
    thrust::device_vector<float> o_device = create_projection<float>(
        batch_size, head_count, sequence_length, head_dim,
        fill_zero_fn
    );

    // Run the naive MHA forward pass
    // Signature: mha_forward(Q, K, V, O, B, H, N, d)
    mha_forward(
        thrust::raw_pointer_cast(q_device.data()),
        thrust::raw_pointer_cast(k_device.data()),
        thrust::raw_pointer_cast(v_device.data()),
        thrust::raw_pointer_cast(o_device.data()),
        batch_size,      // B
        head_count,      // H
        sequence_length, // N
        head_dim         // d
    );

    // Sync and check for CUDA errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}
