#pragma once
#include <cmath>
#include <cuda_runtime.h>

namespace cobraml::kernels {

// kernel 1 (QK^T):
__global__ void qk_kernel(float *Q, float *K, float *S, int B, int N, int H,
                          int d) {
  int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = B * H * N * N;

  if (linear_idx >= total_elements)
    return;

  int b = linear_idx / (H * N * N);
  int remainder = (linear_idx % (H * N * N));
  int h = remainder / (N * N);
  remainder = remainder % (N * N);
  int i = remainder / N; // query position
  int j = remainder % N; // key position

  // compute dot product: Q[b,i,h,:] * K[b,j,h,:]
  float score = 0.0;
  for (int k = 0; k < d; k++) {
    // BSHD: b * (N*H*d) + seq * (H*d) + h * d + k
    int q_idx = b * (N * H * d) + i * (H * d) + h * d + k;
    int k_idx = b * (N * H * d) + j * (H * d) + h * d + k;
    score += Q[q_idx] * K[k_idx];
  }

  // scale by 1/sqrt(d)
  score = score / sqrtf((float)d);

  // write - S is still [B, H, N, N] for convenience
  int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
  S[s_idx] = score;
}

// kernel 2 (Softmax):
__global__ void softmax_kernel(float *S, float *P, int B, int N, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * N;

  if (idx >= total)
    return;

  // decode which row this thread processes
  int b = idx / (H * N);
  int remainder = idx % (H * N);
  int h = remainder / N;
  int i = remainder % N;

  // find max for numerical stability
  float max_val = -INFINITY;
  for (int j = 0; j < N; j++) {
    int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    max_val = fmaxf(max_val, S[s_idx]);
  }

  // compute exp and sum
  float sum = 0.0f;
  for (int j = 0; j < N; j++) {
    int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    float exp_val = expf(S[s_idx] - max_val);
    sum += exp_val;
  }

  // write softmax
  for (int j = 0; j < N; j++) {
    int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    int p_idx = s_idx;
    float exp_val = expf(S[s_idx] - max_val);
    P[p_idx] = exp_val / sum;
  }
}

// kernel 3 (PV):
__global__ void pv_kernel(float *P, float *V, float *O, int B, int N, int H,
                          int d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * H * N * d;

  if (idx >= total)
    return;

  // decode which output element this thread computes
  int b = idx / (H * N * d);
  int remainder = idx % (H * N * d);
  int h = remainder / (N * d);
  remainder = remainder % (N * d);
  int i = remainder / d;
  int k = remainder % d;

  // compute O[b,i,h,k] = sum_j P[b,h,i,j] * V[b,j,h,k]
  float output = 0.0f;
  for (int j = 0; j < N; j++) {
    // P is [B, H, N, N]
    int p_idx = b * (H * N * N) + h * (N * N) + i * N + j;
    // V is BSHD: [B, N, H, d]
    int v_idx = b * (N * H * d) + j * (H * d) + h * d + k;
    output += P[p_idx] * V[v_idx];
  }

  // write - O is BSHD: [B, N, H, d]
  int o_idx = b * (N * H * d) + i * (H * d) + h * d + k;
  O[o_idx] = output;
}

// launcher
void mha_forward(float *Q, float *K, float *V, float *O, int B, int N, int H,
                 int d) {
  // Allocate intermediate buffers - S and P are [B, H, N, N]
  float *S, *P;
  cudaMalloc(&S, B * H * N * N * sizeof(float));
  cudaMalloc(&P, B * H * N * N * sizeof(float));

  // Launch kernels
  int threads = 256;
  qk_kernel<<<(B * H * N * N + threads - 1) / threads, threads>>>(Q, K, S, B, N,
                                                                  H, d);
  softmax_kernel<<<(B * H * N + threads - 1) / threads, threads>>>(S, P, B, N,
                                                                   H);
  pv_kernel<<<(B * H * N * d + threads - 1) / threads, threads>>>(P, V, O, B, N,
                                                                  H, d);

  cudaDeviceSynchronize();
  cudaFree(S);
  cudaFree(P);
}

} // namespace cobraml::kernels
