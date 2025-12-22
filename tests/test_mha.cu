#include <gtest/gtest.h>
#include <cobraml2/kernels/mha.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>

using namespace cobraml::kernels;

template<typename DType>
void fill_zero(DType* data, int length) {
    cudaMemset(data, 0, length * sizeof(DType));
    cudaDeviceSynchronize();
}

void fill_random_uniform(float* data, int length, int seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, data, length);
    curandDestroyGenerator(gen);
    cudaDeviceSynchronize();
}

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

auto seeded_fill_fn(int seed) {
    return [=](float* data, int length) {
        fill_random_uniform(data, length, seed);
    };
}

TEST(MHA_PURE_CUDA_TEST, kernel_launch_no_errors) {
    // Small test case for quick validation
    constexpr int B = 2;        // batch size
    constexpr int H = 4;        // number of heads
    constexpr int N = 32;       // sequence length
    constexpr int d = 64;       // head dimension

    // Create Q, K, V projections with random data
    thrust::device_vector<float> q_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(0)
    );

    thrust::device_vector<float> k_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(1)
    );

    thrust::device_vector<float> v_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(2)
    );

    // Create output tensor initialized to zero
    auto fill_zero_fn = fill_zero<float>;
    thrust::device_vector<float> o_device = create_projection<float>(
        B, H, N, d, fill_zero_fn
    );

    // Run MHA forward pass
    mha_forward(
        thrust::raw_pointer_cast(q_device.data()),
        thrust::raw_pointer_cast(k_device.data()),
        thrust::raw_pointer_cast(v_device.data()),
        thrust::raw_pointer_cast(o_device.data()),
        B, H, N, d
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

TEST(MHA_PURE_CUDA_TEST, kernel_larger_config) {
    constexpr int B = 8;        // batch size
    constexpr int H = 16;       // number of heads
    constexpr int N = 128;      // sequence length
    constexpr int d = 64;       // head dimension

    thrust::device_vector<float> q_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(42)
    );

    thrust::device_vector<float> k_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(43)
    );

    thrust::device_vector<float> v_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(44)
    );

    auto fill_zero_fn = fill_zero<float>;
    thrust::device_vector<float> o_device = create_projection<float>(
        B, H, N, d, fill_zero_fn
    );

    mha_forward(
        thrust::raw_pointer_cast(q_device.data()),
        thrust::raw_pointer_cast(k_device.data()),
        thrust::raw_pointer_cast(v_device.data()),
        thrust::raw_pointer_cast(o_device.data()),
        B, H, N, d
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}

TEST(MHA_PURE_CUDA_TEST, output_not_zero) {
    constexpr int B = 1;
    constexpr int H = 1;
    constexpr int N = 16;
    constexpr int d = 32;

    thrust::device_vector<float> q_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(100)
    );

    thrust::device_vector<float> k_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(101)
    );

    thrust::device_vector<float> v_device = create_projection<float>(
        B, H, N, d, seeded_fill_fn(102)
    );

    auto fill_zero_fn = fill_zero<float>;
    thrust::device_vector<float> o_device = create_projection<float>(
        B, H, N, d, fill_zero_fn
    );

    mha_forward(
        thrust::raw_pointer_cast(q_device.data()),
        thrust::raw_pointer_cast(k_device.data()),
        thrust::raw_pointer_cast(v_device.data()),
        thrust::raw_pointer_cast(o_device.data()),
        B, H, N, d
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

    // Copy output to host and check it's not all zeros
    thrust::host_vector<float> o_host = o_device;

    float sum = 0.0f;
    for (size_t i = 0; i < o_host.size(); ++i) {
        sum += std::abs(o_host[i]);
    }

    ASSERT_GT(sum, 0.0f) << "Output is all zeros - computation did not happen";
}
