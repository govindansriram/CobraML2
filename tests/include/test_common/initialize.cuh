#pragma once

#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace cobraml::test_helpers {

template <typename DType>
void fill_random_uniform_impl(DType *data, int length, int seed,
                              auto assignment_func) {
  curandGenerator_t gen;
  thrust::device_vector<float> temp_dev_vec_float(length);

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(
      gen, thrust::raw_pointer_cast(temp_dev_vec_float.data()), length);
  curandDestroyGenerator(gen);

  thrust::host_vector<float> temp_host_vec_float(temp_dev_vec_float);
  thrust::host_vector<DType> temp_host_vec_dtype(length);

  cudaDeviceSynchronize();
  for (size_t i{0}; i < length; ++i)
    assignment_func(temp_host_vec_dtype[i], temp_host_vec_float[i]);

  cudaMemcpy(data, thrust::raw_pointer_cast(temp_host_vec_dtype.data()),
             sizeof(DType) * length, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

template <typename DType>
void fill_random_uniform(DType *data, int length, int seed) {
  auto fn = [](DType &dst, float src) { dst = DType(src); };

  fill_random_uniform_impl(data, length, seed, fn);
}

template <typename DType> void fill_zero(DType *data, int length) {
  cudaMemset(data, 0, length * sizeof(DType));
  cudaDeviceSynchronize();
}

template <typename DType> auto seeded_fill_random_uniform(int seed) {
  return
      [=](DType *data, int length) { fill_random_uniform(data, length, seed); };
}

template <typename DType>
thrust::device_vector<DType> create_tensor(size_t length, auto fill_fn) {
  thrust::device_vector<DType> device_vec(length);
  fill_fn(thrust::raw_pointer_cast(device_vec.data()), length);
  return device_vec;
}

}; // namespace cobraml::test_helpers
