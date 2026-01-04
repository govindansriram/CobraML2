#include <cobraml2/kernels/mha.cuh>
#include <gtest/gtest.h>
#include <test_common/mha.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cobraml;

template <int head_count, int head_dim>
void test_mha(int batch_size, int sequence_length) {
  // Create Q tensor [B, H, N, d]
  thrust::device_vector<float> q_device{test_helpers::create_projection<float>(
      head_count, head_dim, batch_size, sequence_length,
      test_helpers::seeded_fill_random_uniform<float>(0))};

  thrust::device_vector<float> k_device{test_helpers::create_projection<float>(
      head_count, head_dim, batch_size, sequence_length,
      test_helpers::seeded_fill_random_uniform<float>(1))};

  thrust::device_vector<float> v_device{test_helpers::create_projection<float>(
      head_count, head_dim, batch_size, sequence_length,
      test_helpers::seeded_fill_random_uniform<float>(2))};

  thrust::device_vector<float> o_device{test_helpers::create_projection<float>(
      head_count, head_dim, batch_size, sequence_length,
      test_helpers::fill_zero<float>)};

#ifdef BENCHMARK
  constexpr size_t warmup_iters{2};
  constexpr size_t total_iters{10};
#else
  constexpr size_t warmup_iters{1};
#endif

  for (size_t i{0}; i < warmup_iters; ++i) {
    kernels::mha_forward(thrust::raw_pointer_cast(q_device.data()),
                         thrust::raw_pointer_cast(k_device.data()),
                         thrust::raw_pointer_cast(v_device.data()),
                         thrust::raw_pointer_cast(o_device.data()),
                         batch_size,      // B
                         head_count,      // H
                         sequence_length, // N
                         head_dim         // d
    );
  }

  // Sync and check for CUDA errors
  cudaDeviceSynchronize();

#ifdef BENCHMARK
  float total_time_ms{0};
  float ms;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (size_t i{0}; i < total_iters; ++i) {
    cudaEventRecord(start);
    kernels::mha_forward(thrust::raw_pointer_cast(q_device.data()),
                         thrust::raw_pointer_cast(k_device.data()),
                         thrust::raw_pointer_cast(v_device.data()),
                         thrust::raw_pointer_cast(o_device.data()),
                         batch_size,      // B
                         head_count,      // H
                         sequence_length, // N
                         head_dim         // d
    );
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

  // Calculate GFLOPs
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

  thrust::host_vector<float> q_host = q_device;
  thrust::host_vector<float> k_host = k_device;
  thrust::host_vector<float> v_host = v_device;
  thrust::host_vector<float> o_host = o_device; // GPU output

  // Compute CPU reference
  std::vector<float> o_ref(o_host.size(), 0.0f);

  test_helpers::cpu_mha(q_host.data(), k_host.data(), v_host.data(),
                        o_ref.data(), batch_size, head_count, sequence_length,
                        head_dim);

  std::vector<float> o_vec(o_host.begin(), o_host.end());
  test_helpers::check_output(o_vec, o_ref, batch_size, head_count,
                             sequence_length, head_dim, 1e-4f);
}

TEST(MHA, H16_D64_B56_N128) { test_mha<16, 64>(56, 128); }
