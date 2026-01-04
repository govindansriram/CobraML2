#pragma once
#include "./initialize.cuh"
#include <gtest/gtest.h>

namespace cobraml::test_helpers {

// CPU reference implementation for correctness verification
// Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
// All tensors have shape [B, H, N, d]
void cpu_mha(float *Q, float *K, float *V, float *O, int B, int H, int N,
             int d) {
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      // Get pointers for this (batch, head)
      float *q = Q + (b * H + h) * N * d;
      float *k = K + (b * H + h) * N * d;
      float *v = V + (b * H + h) * N * d;
      float *o = O + (b * H + h) * N * d;

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

template <typename DType>
thrust::device_vector<DType>
create_projection(int head_count, int head_dim, int batch_size,
                  int sequence_length, auto fill_fn) {
  int total_length{head_count * head_dim * batch_size * sequence_length};
  return create_tensor<DType>(total_length, fill_fn);
}

void check_output(const std::vector<float> &result,
                  const std::vector<float> &expected, int b, int h, int N,
                  int d, float tolerance) {
  // float tolerance = 1e-4f;

  ASSERT_EQ(result.size(), expected.size())
      << "expected and result are incomparable, vector lengths are not the "
         "same";

  for (int i = 0; i < expected.size(); i++) {

    int batch{i / (h * N * d)};
    int leftover{i % (h * N * d)};
    int head{leftover / (N * d)};
    leftover = leftover % (N * d);
    int seq{leftover / d};
    int idx{leftover % d};

    ASSERT_NEAR(result[i], expected[i], tolerance)
        << "Mismatch at batch: " << batch << ", head: " << head
        << ", sequence: " << seq << ", idx: " << idx
        << " ----- result=" << result[i] << ", expected=" << expected[i];
  }
}

float calculate_gflops(size_t b, size_t h, size_t N, size_t d, float ms) {
  size_t flops{4 * b * h * N * N * d};
  float seconds = ms / 1000.0f;
  return flops / seconds / 1e9;
}
} // namespace cobraml::test_helpers