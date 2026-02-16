#pragma once
#include "./initialize.cuh"
#include <gtest/gtest.h>

namespace cobraml::test_helpers {

// CPU reference implementation for correctness verification
// Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
// Q, K, V are separate pointers with a configurable seq_stride.
// Output O always uses standard BSHD layout (seq_stride = H*d).
void cpu_mha_impl(float *Q, float *K, float *V, float *O, int B, int N, int H,
                  int d, int seq_stride, bool causal) {
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int i = 0; i < N; i++) {
        float max_score = -INFINITY;
        std::vector<float> scores(N);

        for (int j = 0; j < N; j++) {
          if (causal && j > i) {
            scores[j] = -INFINITY;
            continue;
          }

          float score = 0;
          for (int k_idx = 0; k_idx < d; k_idx++) {
            int q_idx = b * (seq_stride * N) + i * seq_stride + h * d + k_idx;
            int k_idx_full =
                b * (seq_stride * N) + j * seq_stride + h * d + k_idx;
            score += Q[q_idx] * K[k_idx_full];
          }
          score /= sqrtf((float)d);
          scores[j] = score;
          max_score = std::max(max_score, score);
        }

        float sum_exp = 0;
        for (int j = 0; j < N; j++) {
          scores[j] = expf(scores[j] - max_score);
          sum_exp += scores[j];
        }

        for (int k_idx = 0; k_idx < d; k_idx++) {
          float out = 0;
          for (int j = 0; j < N; j++) {
            int v_idx = b * (seq_stride * N) + j * seq_stride + h * d + k_idx;
            out += (scores[j] / sum_exp) * V[v_idx];
          }
          int o_idx = b * (N * H * d) + i * (H * d) + h * d + k_idx;
          O[o_idx] = out;
        }
      }
    }
  }
}

// Separate Q, K, V buffers (standard BSHD layout)
void cpu_mha(float *Q, float *K, float *V, float *O, int B, int N, int H, int d,
             bool causal = false) {
  cpu_mha_impl(Q, K, V, O, B, N, H, d, H * d, causal);
}

// Fused QKV buffer (Q at offset 0, K at H*d, V at 2*H*d per sequence position)
void cpu_mha_contiguous(float *QKV, float *O, int B, int N, int H, int d,
                        bool causal = false) {
  int hd = H * d;
  cpu_mha_impl(QKV, QKV + hd, QKV + 2 * hd, O, B, N, H, d, hd * 3, causal);
}

template <typename DType>
thrust::device_vector<DType>
create_projection(int batch_size, int sequence_length, int head_count,
                  int head_dim, auto fill_fn) {
  int total_length{batch_size * sequence_length * head_count * head_dim};
  return create_tensor<DType>(total_length, fill_fn);
}

void check_output(const std::vector<float> &result,
                  const std::vector<float> &expected, int b, int N, int h,
                  int d, float tolerance) {

  ASSERT_EQ(result.size(), expected.size())
      << "expected and result are incomparable, vector lengths are not the "
         "same";

  for (int i = 0; i < expected.size(); i++) {
    // BSHD: index = batch * (N*h*d) + seq * (h*d) + head * d + idx
    int batch{i / (N * h * d)};
    int leftover{i % (N * h * d)};
    int seq{leftover / (h * d)};
    leftover = leftover % (h * d);
    int head{leftover / d};
    int idx{leftover % d};

    ASSERT_NEAR(result[i], expected[i], tolerance)
        << "Mismatch at batch: " << batch << ", seq: " << seq
        << ", head: " << head << ", idx: " << idx
        << " ----- result=" << result[i] << ", expected=" << expected[i];
  }
}

float calculate_gflops(size_t b, size_t h, size_t N, size_t d, float ms) {
  size_t flops{4 * b * h * N * N * d};
  float seconds = ms / 1000.0f;
  return flops / seconds / 1e9;
}
} // namespace cobraml::test_helpers