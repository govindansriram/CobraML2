#include <gtest/gtest.h>
#include <cobraml2/kernels/mha_cute.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <vector>
#include <cmath>


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

// CPU reference implementation for correctness verification
// Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
// All tensors have shape [B, H, N, d]
void cpu_attention(
    float* Q, float* K, float* V, float* O,
    int B, int H, int N, int d
) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Get pointers for this (batch, head)
            float* q = Q + (b * H + h) * N * d;
            float* k = K + (b * H + h) * N * d;
            float* v = V + (b * H + h) * N * d;
            float* o = O + (b * H + h) * N * d;

            for (int i = 0; i < N; i++) {
                // Step 1: Compute scores for query i against all keys
                float max_score = -INFINITY;
                std::vector<float> scores(N);

                for (int j = 0; j < N; j++) {
                    float score = 0;
                    for (int k_idx = 0; k_idx < d; k_idx++) {
                        score += q[i * d + k_idx] * k[j * d + k_idx];
                    }
                    score /= sqrtf((float)d);
                    scores[j] = score;
                    max_score = std::max(max_score, score);
                }

                // Step 2: Softmax with numerical stability (subtract max)
                float sum_exp = 0;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                // Step 3: Weighted sum of values
                for (int k_idx = 0; k_idx < d; k_idx++) {
                    float out = 0;
                    for (int j = 0; j < N; j++) {
                        out += (scores[j] / sum_exp) * v[j * d + k_idx];
                    }
                    o[i * d + k_idx] = out;
                }
            }
        }
    }
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
    // Copy inputs to host for CPU reference
    thrust::host_vector<float> q_host = q_device;
    thrust::host_vector<float> k_host = k_device;
    thrust::host_vector<float> v_host = v_device;
    thrust::host_vector<float> o_gpu = o_device;  // GPU output
    
    // Compute CPU reference
    int total_output = batch_size * head_count * sequence_length * head_dim;
    std::vector<float> o_cpu(total_output, 0.0f);
    
    cpu_attention(
        q_host.data(), k_host.data(), v_host.data(), o_cpu.data(),
        batch_size, head_count, sequence_length, head_dim
    );
    
    // Compare GPU vs CPU with tolerance
    float max_diff = 0.0f;
    float tolerance = 1e-4f;
    for (int i = 0; i < total_output; i++) {
        float diff = std::fabs(o_gpu[i] - o_cpu[i]);
        max_diff = std::max(max_diff, diff);
        ASSERT_NEAR(o_gpu[i], o_cpu[i], tolerance)
            << "Mismatch at index " << i
            << ": GPU=" << o_gpu[i] << ", CPU=" << o_cpu[i];
    }
    
    std::cout << "Max diff: " << max_diff << std::endl;
}
