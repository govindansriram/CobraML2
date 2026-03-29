#include <cobraml2/kernels/fmha_cc.cuh>
#include <gtest/gtest.h>
#include <test_common/mha.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cobraml;
using namespace cute;

struct BatchEntry {
  int N_q;
  int N_kv;
};

template <int head_count, int head_dim, int B_r, int B_c,
          int q_seq_stride = head_count * head_dim,
          int kv_seq_stride = head_count * head_dim>
void test_fmha_ragged(const std::vector<BatchEntry> &batch) {
  using MHAType = kernels::FMHA<head_count, head_dim, B_r, B_c, float, 128, q_seq_stride, kv_seq_stride>;

  constexpr int hd{head_count * head_dim};
  constexpr bool contiguous{q_seq_stride != hd || kv_seq_stride != hd};

  // build prefix-sum arrays
  std::vector<uint32_t> cu_seqlens_q_host(batch.size() + 1);
  std::vector<uint32_t> cu_seqlens_kv_host(batch.size() + 1);
  std::vector<uint32_t> cu_tiles_q_host(batch.size() + 1);
  cu_seqlens_q_host[0] = 0;
  cu_seqlens_kv_host[0] = 0;
  cu_tiles_q_host[0] = 0;
  for (size_t i{0}; i < batch.size(); i++) {
    ASSERT_GE(batch[i].N_kv, batch[i].N_q)
        << "N_kv must be >= N_q for batch element " << i;
    cu_seqlens_q_host[i + 1] = cu_seqlens_q_host[i] + batch[i].N_q;
    cu_seqlens_kv_host[i + 1] = cu_seqlens_kv_host[i] + batch[i].N_kv;
    cu_tiles_q_host[i + 1] = cu_tiles_q_host[i]
        + static_cast<uint32_t>(cute::ceil_div(batch[i].N_q, B_r));
  }

  uint32_t total_q{cu_seqlens_q_host.back()};
  uint32_t total_kv{cu_seqlens_kv_host.back()};
  uint32_t total_tiles{cu_tiles_q_host.back()};

  // allocate device tensors
  thrust::device_vector<float> qkv_device;
  thrust::device_vector<float> q_device, k_device, v_device;
  float *q_ptr, *k_ptr, *v_ptr;

  if constexpr (contiguous) {
    uint32_t total_tokens = std::max(total_q, total_kv);
    qkv_device = test_helpers::create_tensor<float>(
        total_tokens * 3 * hd,
        test_helpers::seeded_fill_random_uniform<float>(0));
    float *base = thrust::raw_pointer_cast(qkv_device.data());
    q_ptr = base;
    k_ptr = base + hd;
    v_ptr = base + 2 * hd;
  } else {
    q_device = test_helpers::create_projection<float>(
        total_q, head_count, head_dim,
        test_helpers::seeded_fill_random_uniform<float>(0));
    k_device = test_helpers::create_projection<float>(
        total_kv, head_count, head_dim,
        test_helpers::seeded_fill_random_uniform<float>(1));
    v_device = test_helpers::create_projection<float>(
        total_kv, head_count, head_dim,
        test_helpers::seeded_fill_random_uniform<float>(2));
    q_ptr = thrust::raw_pointer_cast(q_device.data());
    k_ptr = thrust::raw_pointer_cast(k_device.data());
    v_ptr = thrust::raw_pointer_cast(v_device.data());
  }

  thrust::device_vector<float> o_device(total_q * hd, 0.0f);
  float *o_ptr = thrust::raw_pointer_cast(o_device.data());

  thrust::device_vector<uint32_t> cu_seqlens_q_dev(cu_seqlens_q_host.begin(), cu_seqlens_q_host.end());
  thrust::device_vector<uint32_t> cu_seqlens_kv_dev(cu_seqlens_kv_host.begin(), cu_seqlens_kv_host.end());
  thrust::device_vector<uint32_t> cu_tiles_q_dev(cu_tiles_q_host.begin(), cu_tiles_q_host.end());

  MHAType mha{};

#ifdef BENCHMARK
  constexpr size_t warmup_iters{2};
  constexpr size_t total_iters{10};
#else
  constexpr size_t warmup_iters{1};
#endif

  for (size_t i{0}; i < warmup_iters; ++i) {
    mha(q_ptr, k_ptr, v_ptr, o_ptr,
        cu_seqlens_q_dev, cu_seqlens_kv_dev, cu_tiles_q_dev,
        total_q, total_kv, total_tiles);
  }
  cudaDeviceSynchronize();

#ifdef BENCHMARK
  float total_time_ms{0};
  float ms;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (size_t i{0}; i < total_iters; ++i) {
    cudaEventRecord(start);
    mha(q_ptr, k_ptr, v_ptr, o_ptr,
        cu_seqlens_q_dev, cu_seqlens_kv_dev, cu_tiles_q_dev,
        total_q, total_kv, total_tiles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    total_time_ms += ms;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

  float average_time_ms{total_time_ms / total_iters};

  size_t total_flops{0};
  for (const auto &entry : batch) {
    total_flops += 4ULL * head_count * entry.N_q * entry.N_kv * head_dim;
  }

  std::cout << "Avg Kernel execution time: " << average_time_ms << " ms\n";
  std::cout << "Achieved performance: "
            << (total_flops / (average_time_ms / 1000.0) / 1e9)
            << " GFLOPs\n";
#else
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
#endif

  // CPU reference
  thrust::host_vector<float> o_host = o_device;
  std::vector<float> o_ref(total_q * hd, 0.0f);

  if constexpr (contiguous) {
    thrust::host_vector<float> qkv_host = qkv_device;
    float *base = qkv_host.data();
    test_helpers::cpu_mha_ragged(base, base + hd, base + 2 * hd, o_ref.data(),
                                 cu_seqlens_q_host, cu_seqlens_kv_host,
                                 head_count, head_dim,
                                 q_seq_stride, kv_seq_stride);
  } else {
    thrust::host_vector<float> q_host = q_device;
    thrust::host_vector<float> k_host = k_device;
    thrust::host_vector<float> v_host = v_device;
    test_helpers::cpu_mha_ragged(q_host.data(), k_host.data(), v_host.data(),
                                 o_ref.data(),
                                 cu_seqlens_q_host, cu_seqlens_kv_host,
                                 head_count, head_dim,
                                 q_seq_stride, kv_seq_stride);
  }

  std::vector<float> o_vec(o_host.begin(), o_host.end());
  constexpr float tolerance{head_dim >= 128 ? 1e-3f : 1e-4f};
  test_helpers::check_output_ragged(o_vec, o_ref, cu_seqlens_q_host,
                                    head_count, head_dim, tolerance);
}

// === Uniform batch (all same length) ===

TEST(FMHA_CC, H16_D64_Br64_Bc64_B4_N512) {
  test_fmha_ragged<16, 64, 64, 64>({{512, 512}, {512, 512}, {512, 512}, {512, 512}});
}

TEST(FMHA_CC, H2_D64_Br64_Bc64_B4_N448_Nkv490) {
  test_fmha_ragged<2, 64, 64, 64>({{448, 490}, {448, 490}, {448, 490}, {448, 490}});
}

TEST(FMHA_CC, H16_D64_Br64_Bc64_B8_N64) {
  test_fmha_ragged<16, 64, 64, 64>({{64, 64}, {64, 64}, {64, 64}, {64, 64},
                                     {64, 64}, {64, 64}, {64, 64}, {64, 64}});
}

TEST(FMHA_CC, H16_D128_Br32_Bc32_B4_N512) {
  test_fmha_ragged<16, 128, 32, 32>({{512, 512}, {512, 512}, {512, 512}, {512, 512}});
}

// === Mixed prefill + decode ===

TEST(FMHA_CC_RAGGED, prefill_and_decode_mixed) {
  test_fmha_ragged<16, 64, 64, 64>({
      {256, 256},   // prefill
      {64, 512},    // decode (1 Q tile, 512 cached KV)
      {128, 128},   // prefill
      {64, 300},    // decode (1 Q tile, 300 cached KV — not divisible by B_c)
  });
}

TEST(FMHA_CC_RAGGED, decode_only_varying_cache) {
  test_fmha_ragged<16, 64, 64, 64>({
      {64, 100},
      {64, 256},
      {64, 67},
      {64, 513},
  });
}

TEST(FMHA_CC_RAGGED, prefill_varying_lengths) {
  test_fmha_ragged<2, 64, 64, 64>({
      {64, 64},
      {128, 128},
      {256, 256},
      {192, 192},
  });
}

TEST(FMHA_CC_RAGGED, decode_single_large_cache) {
  test_fmha_ragged<16, 64, 64, 64>({
      {64, 1000},
  });
}

TEST(FMHA_CC_RAGGED, prefill_and_decode_d128) {
  test_fmha_ragged<16, 128, 32, 32>({
      {128, 128},   // prefill
      {32, 128},    // decode (not divisible by B_c=32)
      {64, 64},     // prefill
      {32, 97},     // decode (very uneven KV)
  });
}

// === Unpadded Q (non-multiple of B_r) ===

TEST(FMHA_CC_RAGGED, unpadded_q_prefill) {
  test_fmha_ragged<16, 64, 64, 64>({
      {50, 50},     // partial last tile (50 % 64 != 0)
      {100, 100},   // partial last tile (100 % 64 != 0)
  });
}

TEST(FMHA_CC_RAGGED, unpadded_q_decode_mixed) {
  test_fmha_ragged<16, 64, 64, 64>({
      {50, 50},     // prefill, partial tile
      {1, 512},     // single-token decode
      {13, 200},    // partial tile, large KV cache
  });
}

TEST(FMHA_CC_RAGGED, single_token_decode) {
  test_fmha_ragged<16, 64, 64, 64>({
      {1, 1},
      {1, 64},
      {1, 500},
  });
}

// === Contiguous QKV buffer tests ===

TEST(FMHA_CC_CONTIGUOUS, uniform_batch) {
  test_fmha_ragged<16, 64, 64, 64, 16*64*3, 16*64*3>({{512, 512}, {512, 512}, {512, 512}, {512, 512}});
}

TEST(FMHA_CC_CONTIGUOUS, prefill_and_decode) {
  test_fmha_ragged<16, 64, 64, 64, 16*64*3, 16*64*3>({
      {256, 256},
      {64, 512},
      {128, 128},
      {64, 300},
  });
}

TEST(FMHA_CC_CONTIGUOUS, d128_mixed) {
  test_fmha_ragged<16, 128, 32, 32, 16*128*3, 16*128*3>({
      {128, 128},
      {32, 200},
      {64, 64},
      {32, 97},
  });
}
