// kernel 1 (QK^T):
    // Grid: (ceil(N/TILE), ceil(N/TILE), B*H)
    // Block: (TILE, TILE)

// kernel 2 (Softmax):
    // Grid: (B*H*N) --> one block per row
    // Block: (min(N, 1024), ) --> or use multiple warps

// kernel 3 (PV):
    // Grid: (ceil(N/TILE), ceil(d/TILE), B*H)
    // Block: (TILE, TILE)

// Output = softmax(Q @ K^T / sqrt(d)) @ V
// Q, K, V: [B, H, N, d] - batch, heads, sequence length, head dimension
// attention matrix S = Q @ K^T has shape [B, H, N, N]
// the output O has shape [B, H, N, d]

#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace cobraml::kernels{
    // kernel 1 (QK^T):
    __global__ void qk_kernel(float * Q, float * K, float * S, int B, int H, int N, int d){
        int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = B * H * N * N;

        if (linear_idx >= total_elements) return;

        int b = linear_idx / (H * N * N);
        int remainder = (linear_idx % (H * N * N));
        int h = remainder / (N * N);
        remainder = remainder % ( N * N);
        int i = remainder / N; // query position
        int j = remainder % N; // key position

        // compute dot product: Q[b,h,i,:] * K[b,h,j,:]
        float score = 0.0;
        for (int k = 0; k < d; k++){
            int q_idx = b*(H*N*d) + h*(N*d) + i*d + k;
            int k_idx = b*(H*N*d) + h*(N*d) + j*d + k;
            score += Q[q_idx] * K[k_idx];
        }
        
        // scale by 1/sqrt(d)
        score = score / sqrt((float)d);

        // write
        int s_idx = b*(H*N*N) + h*(N*N) + i*N + j;
        S[s_idx] = score;

    }

    // kernel 2 (Softmax):
    __global__ void softmax_kernel(float* S, float* P, int B, int H, int N){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * H * N;

        if (idx >= total) return;

        // decode which row this thread processes
        int b = idx / (H * N);
        int remainder = idx % (H * N);
        int h = remainder / N;
        int i = remainder % N;

        // find max for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < N; j++){
            int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
            max_val = fmaxf(max_val, S[s_idx]);
        }

        // compute exp and sum
        float sum = 0.0;
        for (int j = 0; j < N; j++) {
            int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
            float exp_val = expf(S[s_idx] - max_val);
            sum += exp_val;
        }

        // write softmax
        for (int j = 0; j < N; j++) {
            int s_idx = b * (H * N * N) + h * (N * N) + i * N + j;
            int p_idx = b * (H * N * N) + h * (N * N) + i * N + j;
            float exp_val = expf(S[s_idx] - max_val);
            P[p_idx] = exp_val / sum;
        }
    }

    // kernel 3 (PV):
    __global__ void pv_kernel(float* P, float* V, float* O, int B, int H, int N, int d){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * H * N * d;

        if (idx >= total) return;

        // decode which row this thread processes
        int b = idx / (H * N * d);
        int remainder = idx % (H * N * d);
        int h = remainder / (N * d);
        remainder = remainder % (N * d);
        int i = remainder / d;
        int k = remainder % d;

        // compute O[b,h,i,k] = sum_j P[b,h,i,j] * V[b,h,j,k]
        float output = 0.0;
        for (int j = 0; j < N; j++) {
            int p_idx = b * (H * N * N) + h * (N * N) + i * N + j;
            int v_idx = b * (H * N * d) + h * (N * d) + j * d + k;
            output += P[p_idx] * V[v_idx];
        }

        // write
        int o_idx = b * (H * N * d) + h * (N * d) + i * d + k;
        O[o_idx] = output;
    }

    // launcher
    void mha_forward(float* Q, float* K, float* V, float* O, int B, int H, int N, int d) {
        // Allocate intermediate buffers
        float *S, *P;
        cudaMalloc(&S, B * H * N * N * sizeof(float));
        cudaMalloc(&P, B * H * N * N * sizeof(float));

        // Launch kernels
        int threads = 256;
        qk_kernel<<<(B*H*N*N + threads - 1) / threads, threads>>>(Q, K, S, B, H, N, d);
        softmax_kernel<<<(B*H*N + threads - 1) / threads, threads>>>(S, P, B, H, N);
        pv_kernel<<<(B*H*N*d + threads - 1) / threads, threads>>>(P, V, O, B, H, N, d);

        cudaDeviceSynchronize();
        cudaFree(S);
        cudaFree(P);
    }
}