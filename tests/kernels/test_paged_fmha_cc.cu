#include <cobraml2/kernels/paged_fmha_cc.cuh>
#include <gtest/gtest.h>
#include <test_common/mha.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cobraml2/runtime/kv_cache.cuh>
#include <algorithm>
#include <random>
#include <numeric>

using namespace cobraml;
using namespace cute;

// TODO: testing code goes here.

template<
    typename DType,
    typename LayoutType
>
struct OwnedTensor {
    thrust::device_vector<DType> buffer;
    const LayoutType layout;
    OwnedTensor(const thrust::device_vector<DType> &buffer, const LayoutType &layout): buffer(buffer), layout(layout){}

    auto get_tensor() {
        return make_tensor(
              make_gmem_ptr(thrust::raw_pointer_cast(this->buffer.data())), 
              layout
          );
    }

};

auto create_cumulative_vector(const std::vector<int> &vector){
    thrust::host_vector<int> cu_vector(vector.size() + 1, 0);
    for (size_t idx{0}; idx < vector.size(); ++idx)
        cu_vector[idx + 1] = cu_vector[idx] + vector[idx];

    thrust::device_vector<int> device_vector{cu_vector};
    OwnedTensor tensor(device_vector, make_shape(vector.size() + 1));
    return tensor;
}

template<int num_heads, int head_dim>
auto create_qo_vector(const std::vector<int> q_len){
    int N{std::accumulate(q_len.begin(), q_len.end(), 0)};
    auto qo_layout = make_layout(make_shape(N, Int<num_heads>{}, Int<head_dim>{}), LayoutRight{});
    int length{static_cast<int>(size(qo_layout.shape()))};

    thrust::host_vector<float> q_vector_host(length);
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};
    std::mt19937 gen{std::random_device{}()};
    std::generate(q_vector_host.begin(), q_vector_host.end(), [&]{ return dist(gen); });

    thrust::device_vector<float> q_vector{q_vector_host};
    thrust::device_vector<float> o_vector(length);

    OwnedTensor q_tensor(q_vector, qo_layout);
    OwnedTensor o_tensor(o_vector, qo_layout);

    return make_tuple(q_tensor, o_tensor);
}

template<int block_size>
auto create_page_map_tensor(std::vector<int> k_len, size_t num_pages){
    size_t num_requests{k_len.size()};
    int max_it{*std::max_element(k_len.begin(), k_len.end())};
    int max_blocks{(max_it + block_size - 1) / block_size};

    auto page_layout = make_layout(
        make_shape(num_requests, max_blocks), LayoutRight{}
    );

    int num_blocks{static_cast<int>(size(page_layout.shape()))};
    thrust::host_vector<int> host_page_vector(num_blocks, -1);

    Tensor host_page_tensor{
        make_tensor(
            thrust::raw_pointer_cast(host_page_vector.data()), 
            page_layout
        )
    };

    std::vector<int> available_pages(num_pages);
    std::iota(available_pages.begin(), available_pages.end(), 1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(available_pages.begin(), available_pages.end(), gen);

    int counter{0};

    for (size_t i{0}; i < num_requests; ++i){
        for (size_t j{0}; j < (k_len[i] + 16 - 1) / 16; ++j){
            host_page_tensor(i, j) = available_pages[counter];
            ++counter;
        }
    }

    thrust::device_vector<int> device_page_vector{host_page_vector};
    OwnedTensor page_tensor(device_page_vector, page_layout);
    return page_tensor;
}

void test_paged_fmha(std::vector<int>&& q_len_host, std::vector<int>&& kv_len_host, std::vector<int>&& k_new_host) {

    ASSERT_EQ(q_len_host.size(), kv_len_host.size());
    ASSERT_EQ(q_len_host.size(), k_new_host.size());

    int num_requests{static_cast<int>(q_len_host.size())};
    int max_q{*std::max_element(q_len_host.begin(), q_len_host.end())};

    const int num_pages{1000};
    runtime::KVCacheMHA<float, 2, 16, 64, 16> cache(num_pages);
    thrust::host_vector<float> host_cache(cache.elements_per_kv_page * num_pages);

    std::uniform_real_distribution<float> dist{0.0f, 1.0f};
    std::mt19937 gen{std::random_device{}()};
    std::generate(host_cache.begin(), host_cache.end(), [&]{ return dist(gen); });

    cudaMemcpy(
        cache.buffer_tensor.data().get(), 
        host_cache.data(), 
        sizeof(float) * num_pages * cache.elements_per_kv_page, 
        cudaMemcpyHostToDevice
    );

    thrust::device_vector<int> kv_len(kv_len_host);
    // ^ prefill, prefill, prefill, decode, decode, prefill

    auto owned_cu_seqlen_q{create_cumulative_vector(q_len_host)};
    auto owned_cu_seqlen_k_new{create_cumulative_vector(k_new_host)};
    Tensor cache_seqlen{
        make_tensor(
            make_gmem_ptr(thrust::raw_pointer_cast(kv_len.data())), 
            make_shape(kv_len_host.size())
        )
    };

    auto owned_page_map{create_page_map_tensor<16>(kv_len_host, num_pages)};

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

    using ConfigType = kernels::PagedFMHACC_Config<64, 64, 4, false>;
    kernels::PagedFMHACC paged_attention(cache.get_k_cache<1>(), cache.get_v_cache<1>(), owned_page_map.get_tensor(), ConfigType{});

    auto[q_tensor, o_tensor]{create_qo_vector<16, 64>(q_len_host)};
    paged_attention(q_tensor.get_tensor(), o_tensor.get_tensor(), cache_seqlen, owned_cu_seqlen_q.get_tensor(), owned_cu_seqlen_k_new.get_tensor(), num_requests, max_q);
}

TEST(PAGED_FMHA_CC, basic_test){
    test_paged_fmha(
        {32, 10, 16, 1, 1, 5}, 
        {96, 10, 17, 44, 999, 200}, 
        {32, 10, 16, 1, 1, 5}
    );
}