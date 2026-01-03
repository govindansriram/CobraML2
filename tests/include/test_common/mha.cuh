#pragma once
#include "./initialize.cuh"

namespace cobraml::test_helpers{

    // CPU reference implementation for correctness verification
    // Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
    // All tensors have shape [B, H, N, d]
    void cpu_mha(
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

    template<typename DType>
    thrust::device_vector<DType> create_projection(
        int head_count, 
        int head_dim, 
        int batch_size, 
        int sequence_length,
        auto fill_fn
    ){
        int total_length{head_count * head_dim * batch_size * sequence_length};
        return create_tensor<DType>(total_length, fill_fn);
    }

}