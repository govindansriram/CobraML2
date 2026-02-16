#include <cobraml2/kernels/fmha_cc.cuh>
#include <gtest/gtest.h>
#include <test_common/mha.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cobraml;
using namespace cute;

template <int head_count, int head_dim, int B_r, int B_c, bool mask = false,
          bool contiguous = false>
void test_fmha(int batch_size, int sequence_length) {
  using MHAType = kernels::FMHA<head_count, head_dim, B_r, B_c, float, 128,
                                mask, contiguous>;

  constexpr int hd = head_count * head_dim;

  // allocate buffers — fused QKV or separate depending on contiguous flag
  thrust::device_vector<float> qkv_device;
  thrust::device_vector<float> q_device, k_device, v_device;

  if constexpr (contiguous) {
    int total_qkv = batch_size * sequence_length * hd * 3;
    qkv_device = test_helpers::create_tensor<float>(
        total_qkv, test_helpers::seeded_fill_random_uniform<float>(0));
  } else {
    q_device = test_helpers::create_projection<float>(
        head_count, head_dim, batch_size, sequence_length,
        test_helpers::seeded_fill_random_uniform<float>(0));
    k_device = test_helpers::create_projection<float>(
        head_count, head_dim, batch_size, sequence_length,
        test_helpers::seeded_fill_random_uniform<float>(1));
    v_device = test_helpers::create_projection<float>(
        head_count, head_dim, batch_size, sequence_length,
        test_helpers::seeded_fill_random_uniform<float>(2));
  }

  thrust::device_vector<float> o_device{test_helpers::create_projection<float>(
      head_count, head_dim, batch_size, sequence_length,
      test_helpers::fill_zero<float>)};

  // resolve Q, K, V pointers
  float *q_ptr, *k_ptr, *v_ptr;
  if constexpr (contiguous) {
    float *base = thrust::raw_pointer_cast(qkv_device.data());
    q_ptr = base;
    k_ptr = base + hd;
    v_ptr = base + 2 * hd;
  } else {
    q_ptr = thrust::raw_pointer_cast(q_device.data());
    k_ptr = thrust::raw_pointer_cast(k_device.data());
    v_ptr = thrust::raw_pointer_cast(v_device.data());
  }
  float *o_ptr = thrust::raw_pointer_cast(o_device.data());

  MHAType mha{};

#ifdef BENCHMARK
  constexpr size_t warmup_iters{2};
  constexpr size_t total_iters{10};
#else
  constexpr size_t warmup_iters{1};
#endif

  for (size_t i{0}; i < warmup_iters; ++i) {
    mha(q_ptr, k_ptr, v_ptr, o_ptr, batch_size, sequence_length);
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
    mha(q_ptr, k_ptr, v_ptr, o_ptr, batch_size, sequence_length);
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

  std::cout << "Avg Kernel execution time: " << average_time_ms << " ms\n";
  std::cout << "Achieved performance: "
            << test_helpers::calculate_gflops(batch_size, head_count,
                                              sequence_length, head_dim,
                                              average_time_ms)
            << " GFLOPs\n";
#else
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
#endif

  thrust::host_vector<float> o_host = o_device;
  std::vector<float> o_ref(o_host.size(), 0.0f);

  if constexpr (contiguous) {
    thrust::host_vector<float> qkv_host = qkv_device;
    test_helpers::cpu_mha_contiguous(qkv_host.data(), o_ref.data(), batch_size,
                                     sequence_length, head_count, head_dim,
                                     mask);
  } else {
    thrust::host_vector<float> q_host = q_device;
    thrust::host_vector<float> k_host = k_device;
    thrust::host_vector<float> v_host = v_device;
    test_helpers::cpu_mha(q_host.data(), k_host.data(), v_host.data(),
                          o_ref.data(), batch_size, sequence_length, head_count,
                          head_dim, mask);
  }

  std::vector<float> o_vec(o_host.begin(), o_host.end());
  test_helpers::check_output(o_vec, o_ref, batch_size, sequence_length,
                             head_count, head_dim, 1e-4f);
}

// === Separate buffer tests ===

// even block size by sequence length
TEST(FMHA_CC, H16_D64_Br64_Bc64_B4_N512) { test_fmha<16, 64, 64, 64>(4, 512); }

// even block size by sequence length with causal masking
TEST(FMHA_CC, H16_D64_Br64_Bc64_B4_N512_causal) {
  test_fmha<16, 64, 64, 64, true>(4, 512);
}

// uneven block size by sequence length (requires predication)
TEST(FMHA_CC, H2_D64_Br64_Bc64_B56_N490) { test_fmha<2, 64, 64, 64>(56, 490); }

// uneven block size by sequence length (requires predication) with causal
// masking
TEST(FMHA_CC, H2_D64_Br64_Bc64_B56_N490_causal) {
  test_fmha<2, 64, 64, 64, true>(56, 490);
}

// 1 block only and even
TEST(FMHA_CC, H16_D64_Br64_Bc64_B8_N64) { test_fmha<16, 64, 64, 64>(8, 64); }

// 1 block only and even with causal masking
TEST(FMHA_CC, H16_D64_Br64_Bc64_B8_N64_causal) {
  test_fmha<16, 64, 64, 64, true>(8, 64);
}

// 1 block only and uneven
TEST(FMHA_CC, H16_D64_Br64_Bc64_B8_N59) { test_fmha<16, 64, 64, 64>(8, 59); }

// 1 block only and uneven with causal masking
TEST(FMHA_CC, H16_D64_Br64_Bc64_B8_N59_causal) {
  test_fmha<16, 64, 64, 64, true>(8, 59);
}

// even block size by sequence length head_dim 128
TEST(FMHA_CC, H16_D128_Br32_Bc32_B4_N512) {
  test_fmha<16, 128, 32, 32>(4, 512);
}

// even block size by sequence length with causal masking, head_dim 128
TEST(FMHA_CC, H16_D128_Br32_Bc32_B4_N512_causal) {
  test_fmha<16, 128, 32, 32, true>(4, 512);
}

// uneven block size by sequence length (requires predication) head_dim 128
TEST(FMHA_CC, H2_D128_Br32_Bc32_B56_N490) {
  test_fmha<2, 128, 32, 32>(56, 490);
}

// uneven block size by sequence length (requires predication) with causal
// masking head_dim 128
TEST(FMHA_CC, H2_D128_Br32_Bc32_B56_N490_causal) {
  test_fmha<2, 128, 32, 32, true>(56, 490);
}

// === Contiguous QKV buffer tests (simulates split from fused projection) ===

// even block size
TEST(FMHA_CC_CONTIGUOUS, H16_D64_Br64_Bc64_B4_N512) {
  test_fmha<16, 64, 64, 64, false, true>(4, 512);
}

TEST(FMHA_CC_CONTIGUOUS, H16_D64_Br64_Bc64_B4_N512_causal) {
  test_fmha<16, 64, 64, 64, true, true>(4, 512);
}

// uneven block size (requires predication)
TEST(FMHA_CC_CONTIGUOUS, H2_D64_Br64_Bc64_B56_N490) {
  test_fmha<2, 64, 64, 64, false, true>(56, 490);
}

TEST(FMHA_CC_CONTIGUOUS, H2_D64_Br64_Bc64_B56_N490_causal) {
  test_fmha<2, 64, 64, 64, true, true>(56, 490);
}

// 1 block only and even
TEST(FMHA_CC_CONTIGUOUS, H16_D64_Br64_Bc64_B8_N64) {
  test_fmha<16, 64, 64, 64, false, true>(8, 64);
}

TEST(FMHA_CC_CONTIGUOUS, H16_D64_Br64_Bc64_B8_N64_causal) {
  test_fmha<16, 64, 64, 64, true, true>(8, 64);
}

// 1 block only and uneven
TEST(FMHA_CC_CONTIGUOUS, H16_D64_Br64_Bc64_B8_N59) {
  test_fmha<16, 64, 64, 64, false, true>(8, 59);
}

TEST(FMHA_CC_CONTIGUOUS, H16_D64_Br64_Bc64_B8_N59_causal) {
  test_fmha<16, 64, 64, 64, true, true>(8, 59);
}

// head_dim 128
TEST(FMHA_CC_CONTIGUOUS, H16_D128_Br32_Bc32_B4_N512) {
  test_fmha<16, 128, 32, 32, false, true>(4, 512);
}

TEST(FMHA_CC_CONTIGUOUS, H16_D128_Br32_Bc32_B4_N512_causal) {
  test_fmha<16, 128, 32, 32, true, true>(4, 512);
}

// uneven head_dim 128
TEST(FMHA_CC_CONTIGUOUS, H2_D128_Br32_Bc32_B56_N490) {
  test_fmha<2, 128, 32, 32, false, true>(56, 490);
}

TEST(FMHA_CC_CONTIGUOUS, H2_D128_Br32_Bc32_B56_N490_causal) {
  test_fmha<2, 128, 32, 32, true, true>(56, 490);
}
