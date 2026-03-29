#include <cobraml2/kernels/fmha_cc.cuh>
#include <torch/extension.h>
#include <thrust/device_vector.h>

template <int H_start, int H_stop, int H_step, int D_start, int D_stop,
          int D_step>
struct FMHADispatcher {

  static constexpr int count_steps(int start, int stop, int step) {
    return (stop - start) / step + 1;
  }

  static constexpr int H_count{count_steps(H_start, H_stop, H_step)};
  static constexpr int D_count{count_steps(D_start, D_stop, D_step)};

  // times 2 for contiguous vs separate
  static constexpr int num_configs{H_count * D_count * 2};

  struct Config {
    int heads, head_dim;
    bool contiguous;
  };

  static constexpr auto idx_to_config(size_t idx) {
    bool contiguous{(idx / (D_count * H_count)) == 0};
    int remainder{static_cast<int>(idx % (D_count * H_count))};
    int head_count{remainder / D_count * H_step};
    int head_dim{remainder % D_count * D_step};

    return Config{head_count + H_start, head_dim + D_start, contiguous};
  }

  using KernelFn = void (*)(float *, float *, float *, float *,
                            const thrust::device_vector<uint32_t> &,
                            const thrust::device_vector<uint32_t> &,
                            const thrust::device_vector<uint32_t> &,
                            uint32_t, uint32_t, uint32_t);

  struct Entry {
    int heads, head_dim;
    bool contiguous;
    KernelFn fn;
  };

  std::array<Entry, num_configs> table;

  template <std::size_t I> void register_one() {
    constexpr Config config{idx_to_config(I)};
    constexpr int hd{config.heads * config.head_dim};
    constexpr int stride{config.contiguous ? hd * 3 : hd};

    table[I] = {
        config.heads, config.head_dim, config.contiguous,
        [](float *Q, float *K, float *V, float *O,
           const thrust::device_vector<uint32_t> &cu_seqlens_q,
           const thrust::device_vector<uint32_t> &cu_seqlens_kv,
           const thrust::device_vector<uint32_t> &cu_tiles_q,
           uint32_t total_q, uint32_t total_kv, uint32_t total_tiles) {
          constexpr Config inner_config{idx_to_config(I)};
          constexpr int inner_hd{inner_config.heads * inner_config.head_dim};
          constexpr int inner_stride{inner_config.contiguous ? inner_hd * 3 : inner_hd};
          constexpr int B_r{inner_config.head_dim >= 128 ? 32 : 64};
          constexpr int B_c{B_r};
          cobraml::kernels::FMHA<inner_config.heads, inner_config.head_dim, B_r,
                                 B_c, float, 128, inner_stride, inner_stride>{}(
              Q, K, V, O, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q,
              total_q, total_kv, total_tiles);
        }};
  }

  template <std::size_t... Is> void init_impl(std::index_sequence<Is...>) {
    (..., register_one<Is>());
  }

  FMHADispatcher() { init_impl(std::make_index_sequence<num_configs>{}); }

  void dispatch(float *Q, float *K, float *V, float *O, bool contiguous,
                const thrust::device_vector<uint32_t> &cu_seqlens_q,
                const thrust::device_vector<uint32_t> &cu_seqlens_kv,
                const thrust::device_vector<uint32_t> &cu_tiles_q,
                uint32_t total_q, uint32_t total_kv, uint32_t total_tiles,
                int H, int d) {
    for (auto &entry : table) {
      if (entry.heads == H && entry.head_dim == d && entry.contiguous == contiguous) {
        entry.fn(Q, K, V, O, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q,
                 total_q, total_kv, total_tiles);
        return;
      }
    }

    TORCH_CHECK(false, "fmha kernel does not exist for the QKV shape provided");
  }
};

// Wrap a torch int32 CUDA tensor as a thrust device_vector (copies data)
thrust::device_vector<uint32_t> wrap_tensor(const torch::Tensor &t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "tensor must be contiguous CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, "tensor must be int32");
  auto *ptr{reinterpret_cast<uint32_t *>(t.data_ptr<int32_t>())};
  int n{static_cast<int>(t.numel())};
  return thrust::device_vector<uint32_t>(ptr, ptr + n);
}

void fmha_forward_fp32(torch::Tensor &Q, torch::Tensor &K, torch::Tensor &V,
                       torch::Tensor &O, bool contiguous,
                       const torch::Tensor &cu_seqlens_q,
                       const torch::Tensor &cu_seqlens_kv,
                       const torch::Tensor &cu_tiles_q,
                       uint32_t total_tiles,
                       const int64_t H, const int64_t d) {

  static FMHADispatcher<1, 32, 1, 64, 128, 64> dispatcher;

  auto cu_seqlens_q_dev{wrap_tensor(cu_seqlens_q)};
  auto cu_seqlens_kv_dev{wrap_tensor(cu_seqlens_kv)};
  auto cu_tiles_q_dev{wrap_tensor(cu_tiles_q)};

  uint32_t total_q{static_cast<uint32_t>(Q.size(0))};
  uint32_t total_kv{static_cast<uint32_t>(K.size(0))};

  dispatcher.dispatch(Q.data_ptr<float>(), K.data_ptr<float>(),
                      V.data_ptr<float>(), O.data_ptr<float>(), contiguous,
                      cu_seqlens_q_dev, cu_seqlens_kv_dev, cu_tiles_q_dev,
                      total_q, total_kv, total_tiles,
                      static_cast<int>(H), static_cast<int>(d));
}

torch::Tensor fmha_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                           torch::Tensor cu_seqlens_q,
                           torch::Tensor cu_seqlens_kv,
                           torch::Tensor cu_tiles_q,
                           int64_t total_tiles) {

  c10::ScalarType Q_dtype{Q.scalar_type()};
  c10::ScalarType K_dtype{K.scalar_type()};
  c10::ScalarType V_dtype{V.scalar_type()};

  TORCH_CHECK(Q_dtype == K_dtype && Q_dtype == V_dtype,
              "dtypes of all three projections must be the same");

  TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3,
              "Q, K, and V must be rank-3 tensors [total_tokens, H, d]");
  TORCH_CHECK(K.sizes() == V.sizes(), "K and V shapes must match");
  TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(2) == K.size(2),
              "Q and K/V must match on heads and head_dim");

  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_kv.dim() == 1 &&
                  cu_tiles_q.dim() == 1,
              "cu_seqlens and cu_tiles_q must be 1D tensors");
  TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_kv.size(0) &&
                  cu_seqlens_q.size(0) == cu_tiles_q.size(0),
              "cu_seqlens_q, cu_seqlens_kv, and cu_tiles_q must have the same length");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 &&
                  cu_seqlens_kv.scalar_type() == torch::kInt32 &&
                  cu_tiles_q.scalar_type() == torch::kInt32,
              "cu_seqlens and cu_tiles_q must be int32");
  TORCH_CHECK(total_tiles > 0, "total_tiles must be positive");

  torch::Tensor O{torch::empty_like(Q)};

  const int64_t H{Q.size(1)};
  const int64_t d{Q.size(2)};

  // detect contiguous QKV: K starts right after Q's H*d block
  bool contiguous{K.data_ptr() == static_cast<void *>(
      static_cast<char *>(Q.data_ptr()) + H * d * Q.element_size())};

  if (Q_dtype == torch::kFloat32) {
    fmha_forward_fp32(Q, K, V, O, contiguous, cu_seqlens_q, cu_seqlens_kv,
                      cu_tiles_q, static_cast<uint32_t>(total_tiles), H, d);
    return O;
  } else {
    TORCH_CHECK(false, "currently only float32 FMHA is supported");
    return O;
  }
}

TORCH_LIBRARY(cobraml, m) {
  m.def("fmha(Tensor Q, Tensor K, Tensor V, Tensor cu_seqlens_q, "
        "Tensor cu_seqlens_kv, Tensor cu_tiles_q, int total_tiles) -> Tensor");
}

TORCH_LIBRARY_IMPL(cobraml, CUDA, m) { m.impl("fmha", &fmha_forward); }
