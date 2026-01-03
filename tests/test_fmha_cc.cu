#include <cobraml2/kernels/fmha_cc.cuh>
#include <gtest/gtest.h>
#include <test_common/mha.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace cobraml;
using namespace cute;

template <int head_count, int head_dim, int B_r, int B_c>
void test_fmha(int batch_size, int sequence_length) {
  using MHAType = kernels::FMHA<head_count, head_dim, B_r, B_c, float>;

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

  MHAType mha{};

  mha(thrust::raw_pointer_cast(q_device.data()),
      thrust::raw_pointer_cast(k_device.data()),
      thrust::raw_pointer_cast(v_device.data()),
      thrust::raw_pointer_cast(o_device.data()), batch_size, sequence_length);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

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

TEST(FMHA_CC, H16_D64_Br64_Bc64_B56_N128) {
  test_fmha<16, 64, 64, 64>(56, 128);
}