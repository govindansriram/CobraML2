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
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cmath>

namespace cobraml::kernels{

using namespace cute;

// helper functions
template<int TILE_ROW, int TILE_COL, typename DType>
CUTE_HOST_DEVICE
auto make_gemm_tiled_copy() {
    return make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DType>{},
        Layout<Shape<Int<TILE_ROW>, Int<16>>>{}
    );
}

template<int TILE_M, int TILE_N, typename DType>
CUTE_HOST_DEVICE
auto make_gemm_tiled_mma() {
    return make_tiled_mma(
        UniversalFMA<DType, DType, DType>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_N>, _1>>{}
    );
}

namespace mha_cute{
    template<int TILE_N, int TILE_D, typename DType, typename TiledCopyQ, typename TiledCopyK, typename TiledMMA>
    __global__ void qk_kernel(
        const DType * __restrict__ Q, // [B, H, N, d]
        const DType * __restrict__ K, // [B, H, N, d]
        DType* __restrict__ S, // [B, H, N, N]
        int B, int H, int N, int d,
        TiledCopyQ tiled_copy_q,
        TiledCopyK tiled_copy_k,
        TiledMMA tiled_mma
    ){
        // Create global memory tensor views with proper strides
        auto Q_tensor = make_tensor(make_gmem_ptr(Q), make_layout(make_shape(B, H, N, d), make_stride(H*N*d, N*d, d, 1)));
        auto K_tensor = make_tensor(make_gmem_ptr(K), make_layout(make_shape(B, H, N, d), make_stride(H*N*d, N*d, d, 1)));
        auto S_tensor = make_tensor(make_gmem_ptr(S), make_layout(make_shape(B, H, N, N), make_stride(H*N*N, N*N, N, 1)));

        // Decode batch and head indices from blockIdx.z
        int bh = blockIdx.z;
        int b_idx = bh / H;
        int h_idx = bh % H;

        // Slice tensors to current batch and head
        auto Q_bh = Q_tensor(b_idx, h_idx, _, _); // [N, d]
        auto K_bh = K_tensor(b_idx, h_idx, _, _); // [N, d]
        auto S_bh = S_tensor(b_idx, h_idx, _, _); // [N, N]

        // CTA tile coordinates
        int tile_row = blockIdx.y;
        int tile_col = blockIdx.x;

        // Extract this CTA's tile of Q, K, and S
        auto gQ = local_tile(Q_bh,
            make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
            make_coord(tile_row, 0),
            Step<_1, _1>{}
        );

        auto gK = local_tile(K_bh,
            make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
            make_coord(tile_col, 0),
            Step<_1, _1>{}
        );

        auto gS = local_tile(S_bh,
            make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
            make_coord(tile_row, tile_col),
            Step<_1, _1>{}
        );
        
        // Shared memory layouts
        auto sQ_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_D>{}), make_stride(Int<TILE_D>{}, _1{}));
        // sK_load: row-major [TILE_N, TILE_D] for loading K from global memory
        auto sK_load_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_D>{}), make_stride(Int<TILE_D>{}, _1{}));
        auto sK_mma_layout = make_layout(make_shape(Int<TILE_D>{}, Int<TILE_N>{}), make_stride(_1{}, Int<TILE_D>{}));

        __shared__ DType smem_q[cosize_v<decltype(sQ_layout)>];
        __shared__ DType smem_k[cosize_v<decltype(sK_load_layout)>];
        
        // Create shared memory tensor views
        Tensor sQ = make_tensor(make_smem_ptr(smem_q), sQ_layout);
        Tensor sK_load = make_tensor(make_smem_ptr(smem_k), sK_load_layout);
        Tensor sK = make_tensor(make_smem_ptr(smem_k), sK_mma_layout);

        // Thread partitioning for copy operations
        ThrCopy thr_copy_q = tiled_copy_q.get_slice(threadIdx.x);
        ThrCopy thr_copy_k = tiled_copy_k.get_slice(threadIdx.x);

        // partition source (global) tensors for this thread
        Tensor tQgQ = thr_copy_q.partition_S(gQ);
        Tensor tKgK = thr_copy_k.partition_S(gK);
        
        // partition destination (shared) tensors for this thread
        Tensor tQsQ = thr_copy_q.partition_D(sQ);
        Tensor tKsK = thr_copy_k.partition_D(sK_load);

        // thread partitioning for mma operations
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

        // partition shared memory for mma consumption
        Tensor tCsQ = thr_mma.partition_A(sQ);
        Tensor tCsK = thr_mma.partition_B(sK);
        Tensor tCgS = thr_mma.partition_C(gS);

        // allocate and clear accumulator fragment
        Tensor tCrS = thr_mma.make_fragment_C(tCgS);
        clear(tCrS);

        // k-dimension loop: accumulate partial products
        int num_k_tiles = (d + TILE_D - 1) / TILE_D;

        for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
            int k_tile_idx = k_tile;
            
            // get q tile for this k iteration
            auto gQ_k = local_tile(Q_bh,
                make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
                make_coord(tile_row, k_tile_idx),
                Step<_1, _1>{}
            );
            
            // get k tile for this k iteration
            auto gK_k = local_tile(K_bh,
                make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
                make_coord(tile_col, k_tile_idx),
                Step<_1, _1>{}
            );
            
            // partition for this iteration
            Tensor tQgQ_k = thr_copy_q.partition_S(gQ_k);
            Tensor tKgK_k = thr_copy_k.partition_S(gK_k);

            // load q and k tiles to shared memory
            copy(tiled_copy_q, tQgQ_k, tQsQ);
            copy(tiled_copy_k, tKgK_k, tKsK);

            __syncthreads();

            // compute partial gemm: accumulate Q @ K^T
            gemm(tiled_mma, tCsQ, tCsK, tCrS);

            __syncthreads();
        }

        // apply scaling factor 1/sqrt(d) and write to global memory
        DType scale = DType(1.0) / sqrt(DType(d));
        for (int i = 0; i < size(tCrS); ++i) {
            tCrS[i] *= scale;
        }
        copy(tCrS, tCgS);
    }

    template<int BLOCK_SIZE, typename DType>
    __global__ void softmax_kernel(
        const DType* __restrict__ S, // [B, H, N, N]
        DType* __restrict__ P, // [B, H, N, N]
        int B, int H, int N
    ){
        // each block handles one row
        int row_idx = blockIdx.x;
        int total_rows = B * H * N;
        if (row_idx >= total_rows) return;

        // decode (b, h, i) from flattened row index
        int b = row_idx / (H * N);
        int rem = row_idx % (H * N);
        int h = rem / N;
        int i = rem % N;

        // create tensor views
        auto S_tensor = make_tensor(make_gmem_ptr(S), make_layout(make_shape(B, H, N, N), make_stride(H*N*N, N*N, N, 1)));
        auto P_tensor = make_tensor(make_gmem_ptr(P), make_layout(make_shape(B, H, N, N), make_stride(H*N*N, N*N, N, 1)));

        // get this row
        auto S_row = S_tensor(b, h, i, _);
        auto P_row = P_tensor(b, h, i, _);

        // shared memory for parallel reduction
        __shared__ DType smax[BLOCK_SIZE];
        __shared__ DType ssum[BLOCK_SIZE];

        int tid = threadIdx.x;

        // find row maximum
        DType thread_max = -INFINITY;
        for (int j = tid; j < N; j += BLOCK_SIZE) {
            thread_max = fmaxf(thread_max, S_row(j));
        }
        smax[tid] = thread_max;
        __syncthreads();

        // parallel reduction to find global max
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
            }
            __syncthreads();
        }
        DType row_max = smax[0];

        // compute exp(x - max) and sum
        DType thread_sum = DType(0);
        for (int j = tid; j < N; j += BLOCK_SIZE) {
            thread_sum += expf(S_row(j) - row_max);
        }
        ssum[tid] = thread_sum;
        __syncthreads();

        // parallel reduction to find sum
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                ssum[tid] += ssum[tid + stride];
            }
            __syncthreads();
        }
        DType row_sum = ssum[0];

        // write normalized softmax values
        for (int j = tid; j < N; j += BLOCK_SIZE) {
            P_row(j) = expf(S_row(j) - row_max) / row_sum;
        }
    }

    template<int TILE_N, int TILE_D, typename DType, typename TiledCopyP, typename TiledCopyV, typename TiledMMA>
    __global__ void pv_kernel(
        const DType* __restrict__ P, // [B, H, N, N]
        const DType* __restrict__ V, // [B, H, N, d]
        DType* __restrict__ O, // [B, H, N, d]
        int B, int H, int N, int d,
        TiledCopyP tiled_copy_p,
        TiledCopyV tiled_copy_v,
        TiledMMA tiled_mma
    ){
        // create global memory tensor views
        auto P_tensor = make_tensor(make_gmem_ptr(P), make_layout(make_shape(B, H, N, N), make_stride(H*N*N, N*N, N, 1)));
        auto V_tensor = make_tensor(make_gmem_ptr(V), make_layout(make_shape(B, H, N, d), make_stride(H*N*d, N*d, d, 1)));
        auto O_tensor = make_tensor(make_gmem_ptr(O), make_layout(make_shape(B, H, N, d), make_stride(H*N*d, N*d, d, 1)));

        // decode batch and head indices
        int bh = blockIdx.z;
        int b_idx = bh / H;
        int h_idx = bh % H;

        // slice to current batch and head
        auto P_bh = P_tensor(b_idx, h_idx, _, _);  // [N, N]
        auto V_bh = V_tensor(b_idx, h_idx, _, _);  // [N, d]
        auto O_bh = O_tensor(b_idx, h_idx, _, _);  // [N, d]

        // cta tile coordinates
        int tile_row = blockIdx.y;
        int tile_col = blockIdx.x;

        // shared memory layouts
        auto sP_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_N>{}), make_stride(Int<TILE_N>{}, _1{}));
        auto sV_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_D>{}), make_stride(Int<TILE_D>{}, _1{}));

        __shared__ DType smem_p[cosize_v<decltype(sP_layout)>];
        __shared__ DType smem_v[cosize_v<decltype(sV_layout)>];

        Tensor sP = make_tensor(make_smem_ptr(smem_p), sP_layout);
        Tensor sV = make_tensor(make_smem_ptr(smem_v), sV_layout);

        // thread partitioning for copy
        ThrCopy thr_copy_p = tiled_copy_p.get_slice(threadIdx.x);
        ThrCopy thr_copy_v = tiled_copy_v.get_slice(threadIdx.x);

        // thread partitioning for mma
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

        // number of tiles along the reduction dimension
        int num_j_tiles = (N + TILE_N - 1) / TILE_N;

        // get output tile and allocate accumulator
        auto gO = local_tile(O_bh,
            make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
            make_coord(tile_row, tile_col),
            Step<_1, _1>{}
        );
        
        Tensor tCgO = thr_mma.partition_C(gO);
        Tensor tCrO = thr_mma.make_fragment_C(tCgO);
        clear(tCrO);

        // accumulation loop over N dimension
        for (int j_tile = 0; j_tile < num_j_tiles; ++j_tile) {
            // get P tile: [TILE_N, TILE_N]
            auto gP = local_tile(P_bh,
                make_shape(Int<TILE_N>{}, Int<TILE_N>{}),
                make_coord(tile_row, j_tile),
                Step<_1, _1>{}
            );
            
            // get V tile: [TILE_N, TILE_D]
            auto gV = local_tile(V_bh,
                make_shape(Int<TILE_N>{}, Int<TILE_D>{}),
                make_coord(j_tile, tile_col),
                Step<_1, _1>{}
            );
            
            // Partition for copy
            Tensor tPgP = thr_copy_p.partition_S(gP);
            Tensor tVgV = thr_copy_v.partition_S(gV);
            
            Tensor tPsP = thr_copy_p.partition_D(sP);
            Tensor tVsV = thr_copy_v.partition_D(sV);

            // Load P and V tiles to shared memory
            copy(tiled_copy_p, tPgP, tPsP);
            copy(tiled_copy_v, tVgV, tVsV);

            __syncthreads();

            // Partition shared memory for MMA
            Tensor tCsP = thr_mma.partition_A(sP);
            Tensor tCsV = thr_mma.partition_B(sV);

            // Compute P @ V and accumulate
            gemm(tiled_mma, tCsP, tCsV, tCrO);

            __syncthreads();
        }
        
        copy(tCrO, tCgO);
    }

    template<int TILE_N = 16, int TILE_D = 16, int SOFTMAX_BLOCK = 256, typename DType = float>
    void mha_forward(DType* Q, DType* K, DType* V, DType* O, int B, int H, int N, int d) {
        // allocate intermediate buffers
        DType *S, *P;
        cudaMalloc(&S, B * H * N * N * sizeof(DType));
        cudaMalloc(&P, B * H * N * N * sizeof(DType));
        
        {
            auto tiled_copy_q = make_gemm_tiled_copy<TILE_N, TILE_D, DType>();
            auto tiled_copy_k = make_gemm_tiled_copy<TILE_N, TILE_D, DType>();
            auto tiled_mma = make_gemm_tiled_mma<TILE_N, TILE_N, DType>();
            
            constexpr int num_threads = TILE_N * TILE_N;
            
            dim3 grid(
                (N + TILE_N - 1) / TILE_N,
                (N + TILE_N - 1) / TILE_N,
                B * H
            );
            dim3 block(num_threads);
            
            qk_kernel<TILE_N, TILE_D, DType><<<grid, block>>>(
                Q, K, S, B, H, N, d,
                tiled_copy_q, tiled_copy_k, tiled_mma
            );
        }
        
        {
            int total_rows = B * H * N;
            softmax_kernel<SOFTMAX_BLOCK, DType><<<total_rows, SOFTMAX_BLOCK>>>(S, P, B, H, N);
        }
        
        {
            auto tiled_copy_p = make_gemm_tiled_copy<TILE_N, TILE_N, DType>();
            auto tiled_copy_v = make_gemm_tiled_copy<TILE_N, TILE_D, DType>();
            auto tiled_mma = make_gemm_tiled_mma<TILE_N, TILE_D, DType>();
            
            constexpr int num_threads = TILE_N * TILE_D;
            
            dim3 grid(
                (d + TILE_D - 1) / TILE_D,
                (N + TILE_N - 1) / TILE_N,
                B * H
            );
            dim3 block(num_threads);
            
            pv_kernel<TILE_N, TILE_D, DType><<<grid, block>>>(
                P, V, O, B, H, N, d,
                tiled_copy_p, tiled_copy_v, tiled_mma
            );
        }
        
        cudaDeviceSynchronize();
        cudaFree(S);
        cudaFree(P);
    }
}
}