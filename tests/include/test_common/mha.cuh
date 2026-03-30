#pragma once
#include "./initialize.cuh"
#include <gtest/gtest.h>

namespace cobraml::test_helpers {

// CPU reference for ragged-batched MHA with causal masking.
// flat_seq_q / flat_seq_kv are prefix-sum arrays: [0, N_q_0, N_q_0+N_q_1, ...]
// Q/K/V are laid out as (total_tokens, H, d) with the given seq_stride.
// O always uses stride H*d (never interleaved).
void cpu_mha_ragged(float *Q, float *K, float *V, float *O,
                    const std::vector<uint32_t> &flat_seq_q,
                    const std::vector<uint32_t> &flat_seq_kv,
                    int H, int d,
                    size_t seq_stride_q, size_t seq_stride_kv) {

  int batch_size = static_cast<int>(flat_seq_q.size()) - 1;

  for (int b = 0; b < batch_size; b++) {
    int q_start = flat_seq_q[b];
    int q_end = flat_seq_q[b + 1];
    int N_q = q_end - q_start;

    int kv_start = flat_seq_kv[b];
    int kv_end = flat_seq_kv[b + 1];
    int N_kv = kv_end - kv_start;

    for (int h = 0; h < H; h++) {
      for (int i = 0; i < N_q; i++) {
        float max_score = -INFINITY;
        std::vector<float> scores(N_kv);

        // causal bound: Q row i sits at absolute position (N_kv - N_q + i)
        int causal_bound = N_kv - N_q + i + 1;

        for (int j = 0; j < N_kv; j++) {
          if (j >= causal_bound) {
            scores[j] = -INFINITY;
            continue;
          }

          float score = 0;
          for (int k_idx = 0; k_idx < d; k_idx++) {
            int q_idx = (q_start + i) * seq_stride_q + h * d + k_idx;
            int k_idx_full = (kv_start + j) * seq_stride_kv + h * d + k_idx;
            score += Q[q_idx] * K[k_idx_full];
          }
          score /= sqrtf((float)d);
          scores[j] = score;
          max_score = std::max(max_score, score);
        }

        float sum_exp = 0;
        for (int j = 0; j < N_kv; j++) {
          scores[j] = expf(scores[j] - max_score);
          sum_exp += scores[j];
        }

        for (int k_idx = 0; k_idx < d; k_idx++) {
          float out = 0;
          for (int j = 0; j < N_kv; j++) {
            int v_idx = (kv_start + j) * seq_stride_kv + h * d + k_idx;
            out += (scores[j] / sum_exp) * V[v_idx];
          }
          int o_idx = (q_start + i) * (H * d) + h * d + k_idx;
          O[o_idx] = out;
        }
      }
    }
  }
}

template <typename DType>
thrust::device_vector<DType>
create_projection(int total_tokens, int head_count, int head_dim, auto fill_fn) {
  int total_length{total_tokens * head_count * head_dim};
  return create_tensor<DType>(total_length, fill_fn);
}

void check_output_ragged(const std::vector<float> &result,
                         const std::vector<float> &expected,
                         const std::vector<uint32_t> &flat_seq_q,
                         int h, int d,
                         float tolerance) {

  int o_stride{h * d};

  for (int b = 0; b < static_cast<int>(flat_seq_q.size()) - 1; b++) {
    int q_start = flat_seq_q[b];
    int q_end = flat_seq_q[b + 1];

    for (int seq = q_start; seq < q_end; seq++) {
      for (int head = 0; head < h; head++) {
        for (int idx = 0; idx < d; idx++) {
          int i = seq * o_stride + head * d + idx;
          ASSERT_NEAR(result[i], expected[i], tolerance)
              << "Mismatch at batch: " << b
              << ", seq: " << (seq - q_start)
              << ", head: " << head << ", idx: " << idx
              << " ----- result=" << result[i]
              << ", expected=" << expected[i];
        }
      }
    }
  }
}

float calculate_gflops(size_t b, size_t h, size_t N, size_t d, float ms) {
  size_t flops{4 * b * h * N * N * d};
  float seconds = ms / 1000.0f;
  return flops / seconds / 1e9;
}
} // namespace cobraml::test_helpers
