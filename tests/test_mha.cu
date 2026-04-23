#include <cobraml2/kernels/mha.cuh>
#include <gtest/gtest.h>
#include <test_common/mha.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <numeric>


template<size_t warps_per_block>
__global__ void prefix_scan_kernel(
    const int * in, 
    int * out, 
    int * block_prefix,
    int n){

    // one block one warp prefix scan

    size_t lane_id{threadIdx.x % 32};
    size_t warp_id{threadIdx.x / 32};
    size_t global_thread_idx{blockDim.x * blockIdx.x + threadIdx.x};

    size_t total_threads{gridDim.x * warps_per_block * 32};

    int val;

    __shared__ int reduction_space[warps_per_block]; 
    __shared__ int last_prefix;

    if (threadIdx.x == 0){
        last_prefix = 0;
    }

    size_t adjusted_n {(n + total_threads - 1) / total_threads * total_threads};

    for (size_t idx{global_thread_idx}; idx < adjusted_n; idx += total_threads){
        val = 0;
        if (idx < n){
            if (threadIdx.x == 0){
                val += last_prefix;
            }

            val += in[idx];
        }
        
        #pragma unroll
        for (int up_jump{1}; up_jump <= 16; up_jump = up_jump << 1){
            int temp{__shfl_up_sync(0xffffffff, val, up_jump)};
            if (lane_id >= up_jump){
                val += temp;
            }
        }

        if (lane_id == 31){
            reduction_space[warp_id] = val;
        }

        __syncthreads();

        int current_prefix{reduction_space[lane_id]};
        int warp_prefix{current_prefix};

        #pragma unroll
        for (int up_jump{1}; up_jump <= 16; up_jump = up_jump << 1){
            int temp{__shfl_up_sync(0xffffffff, warp_prefix, up_jump)};
            if (lane_id >= up_jump){
                warp_prefix += temp;
            }
        }

        val += __shfl_sync(0xffffffff, warp_prefix - current_prefix, warp_id);

        if (warp_id == (warps_per_block - 1 ) && lane_id == 31)
            last_prefix = val;

        if (idx < n){
            out[idx] = val;
            // if (idx < n && block_prefix == nullptr){
            //     printf("%d \n", out[idx]);
            // }
        }

        __syncthreads();
    }

    if (threadIdx.x == warps_per_block * 32 - 1 && (block_prefix != nullptr)){
        if (blockIdx.x + 1 != gridDim.x){
            block_prefix[blockIdx.x + 1] = val;
        }

        if (blockIdx.x == 0){
            block_prefix[blockIdx.x] = 0;
        }

        // printf("%d \n", block_prefix[blockIdx.x]);
    }
}

template<size_t warps_per_block>
__global__ void prefix_add(
    int * out, 
    int * block_prefix,
    int n){

        int prefix{block_prefix[blockIdx.x]};

        size_t global_thread_idx{blockDim.x * blockIdx.x + threadIdx.x};

        size_t total_threads{gridDim.x * warps_per_block * 32};

        size_t adjusted_n {(n + total_threads - 1) / total_threads * total_threads};

        for (size_t idx{global_thread_idx}; idx < adjusted_n; idx += total_threads){
            if (idx < n){
                // if (threadIdx.x == 0){
                //     printf("blockIdx.x: %d value: %d \n", blockIdx.x, prefix);
                // }
                out[idx] += prefix;
            }
        }
    }


void prefix_scan(const int * in, int * out, int n){

    constexpr size_t num_warps{32};
    constexpr size_t block_dim{num_warps * 32};
    int num_blocks{(n + block_dim - 1) / block_dim};
    num_blocks = min(56, num_blocks);

    int *block_prefixes_in;
    int *block_prefixes_out;
    cudaMalloc(&block_prefixes_in, 4 * num_blocks);
    cudaMalloc(&block_prefixes_out, 4 * num_blocks);

    prefix_scan_kernel<num_warps><<<num_blocks, block_dim>>>(in, out, block_prefixes_in, n);
    prefix_scan_kernel<num_warps><<<1, block_dim>>>(block_prefixes_in, block_prefixes_out, nullptr, num_blocks);
    prefix_add<num_warps><<<num_blocks, block_dim>>>(out, block_prefixes_out, n);

    cudaFree(block_prefixes_out);
    cudaFree(block_prefixes_in);
}


TEST(WarpScan, AllOnes) {
    thrust::host_vector<int> in(32, 1);
    thrust::host_vector<int> expected(32);
    std::iota(expected.begin(), expected.end(), 1);  // [1,2,3,...,32]

    thrust::device_vector<int> d_in = in;
    thrust::device_vector<int> d_out(32);
    prefix_scan(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        32);
    cudaDeviceSynchronize();

    thrust::host_vector<int> out = d_out;
    EXPECT_EQ(out, expected);
}

TEST(WarpScan, TriangularNumbers) {
    thrust::host_vector<int> in(32);
    std::iota(in.begin(), in.end(), 0);  // [0,1,2,...,31]

    thrust::host_vector<int> expected(32);
    for (int i = 0; i < 32; i++) expected[i] = i * (i + 1) / 2;

    thrust::device_vector<int> d_in = in;
    thrust::device_vector<int> d_out(32);
    prefix_scan(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        32);
    cudaDeviceSynchronize();

    thrust::host_vector<int> out = d_out;
    EXPECT_EQ(out, expected);
}

TEST(WarpScan, HandTrace) {
    // [3,1,4,1,5,9,2,6] scans to [3,4,8,9,14,23,25,31]
    thrust::host_vector<int> in(32, 0);
    int vals[8] = {3, 1, 4, 1, 5, 9, 2, 6};
    for (int i = 0; i < 8; i++) in[i] = vals[i];

    thrust::host_vector<int> expected(32, 31);  // lanes 8+ carry 31 from zeros
    int exp8[8] = {3, 4, 8, 9, 14, 23, 25, 31};
    for (int i = 0; i < 8; i++) expected[i] = exp8[i];

    thrust::device_vector<int> d_in = in;
    thrust::device_vector<int> d_out(32);
    prefix_scan(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        32);
    cudaDeviceSynchronize();

    thrust::host_vector<int> out = d_out;
    EXPECT_EQ(out, expected);
}

TEST(WarpScan, MediumSize1000) {
    // One block's worth — exercises multi-warp coordination
    const int N = 1000;
    thrust::host_vector<int> in(N);
    for (int i = 0; i < N; i++) in[i] = (i * 13 + 7) % 23 - 11;

    thrust::host_vector<int> expected(N);
    std::partial_sum(in.begin(), in.end(), expected.begin());

    thrust::device_vector<int> d_in = in;
    thrust::device_vector<int> d_out(N);
    prefix_scan(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        N);
    cudaDeviceSynchronize();

    thrust::host_vector<int> out = d_out;

    for (int i = 0; i < N; i++) {
        ASSERT_EQ(out[i], expected[i])
            << "Mismatch at index " << i
            << " (input[i]=" << in[i] << ")";
    }
}

TEST(WarpScan, LargeSize100k) {
    // Exercises cross-block coordination
    const int N = 100'000;
    thrust::host_vector<int> in(N);
    for (int i = 0; i < N; i++) in[i] = (i % 7) - 3;  // small values to avoid overflow

    thrust::host_vector<int> expected(N);
    std::partial_sum(in.begin(), in.end(), expected.begin());

    thrust::device_vector<int> d_in = in;
    thrust::device_vector<int> d_out(N);
    prefix_scan(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        N);
    cudaDeviceSynchronize();

    thrust::host_vector<int> out = d_out;
    EXPECT_EQ(out, expected);
}

// TEST(WarpScan, LargeSize1M) {
//     // Stress test: 1 million elements
//     const int N = 1'000'000;
//     thrust::host_vector<int> in(N, 1);  // all ones → expected[i] = i+1

//     thrust::host_vector<int> expected(N);
//     std::iota(expected.begin(), expected.end(), 1);

//     thrust::device_vector<int> d_in = in;
//     thrust::device_vector<int> d_out(N);
//     prefix_scan(
//         thrust::raw_pointer_cast(d_in.data()),
//         thrust::raw_pointer_cast(d_out.data()),
//         N);
//     cudaDeviceSynchronize();

//     thrust::host_vector<int> out = d_out;
//     EXPECT_EQ(out, expected);
// }

// TEST(WarpScan, NonPowerOfTwo) {
//     // Catches edge cases with partial final block
//     const int N = 12345;
//     thrust::host_vector<int> in(N);
//     for (int i = 0; i < N; i++) in[i] = (i % 5) - 2;

//     thrust::host_vector<int> expected(N);
//     std::partial_sum(in.begin(), in.end(), expected.begin());

//     thrust::device_vector<int> d_in = in;
//     thrust::device_vector<int> d_out(N);
//     prefix_scan(
//         thrust::raw_pointer_cast(d_in.data()),
//         thrust::raw_pointer_cast(d_out.data()),
//         N);
//     cudaDeviceSynchronize();

//     thrust::host_vector<int> out = d_out;
//     EXPECT_EQ(out, expected);
// }