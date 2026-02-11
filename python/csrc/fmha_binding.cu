#include <cobraml2/kernels/fmha_cc.cuh>
#include <torch/extension.h>

template <int H_start, int H_stop, int H_step, int D_start, int D_stop,
          int D_step>
struct FMHADispatcher {

  // Helper to count steps
  static constexpr int count_steps(int start, int stop, int step) {
    return (stop - start) / step + 1;
  }

  static constexpr int H_count{count_steps(H_start, H_stop, H_step)};
  static constexpr int D_count{count_steps(D_start, D_stop, D_step)};

  // times 2 for causal masking
  static constexpr int num_configs{H_count * D_count * 2};

  struct Config {
    int heads, head_dim;
    bool causal;
  };

  static constexpr auto idx_to_config(size_t idx) {

    bool causal{(idx / (D_count * H_count)) == 0};
    int remainder{(idx % (D_count * H_count))};
    int head_count{remainder / D_count * H_step};
    int head_dim{remainder % D_count * D_step};

    return Config{head_count + H_start, head_dim + D_start, causal};
  }

  using KernelFn = void (*)(float *, float *, float *, float *, uint32_t,
                            uint32_t);

  struct Entry {
    int heads, head_dim;
    bool causal;
    KernelFn fn;
  };

  std::array<Entry, num_configs> table;

  template <std::size_t I> void register_one() {
    constexpr Config config{idx_to_config(I)};
    table[I] = {
        config.heads, config.head_dim, config.causal,
        [](float *Q, float *K, float *V, float *O, uint32_t B, uint32_t N) {
          constexpr Config inner_config{idx_to_config(I)};
          cobraml::kernels::FMHA<inner_config.heads, inner_config.head_dim, 64, 64, float, 128,
                                 inner_config.causal>{}(Q, K, V, O, B, N);
        }};
  }

  template <std::size_t... Is> void init_impl(std::index_sequence<Is...>) {
    (..., register_one<Is>());
  }

  FMHADispatcher() { init_impl(std::make_index_sequence<num_configs>{}); }

  void dispatch(float *Q, float *K, float *V, float *O, bool causal, uint32_t B,
                uint32_t N, int H, int d) {
    for (auto &entry : table) {
      if (entry.heads == H && entry.head_dim == d && entry.causal == causal) {
        entry.fn(Q, K, V, O, B, N);
        return;
      }
    }

    TORCH_CHECK(false, "fmha kernel does not exist for the QKV shape provided");
  }
};

// dispatches to flash attention supports all hardware
void fmha_forward_fp32(torch::Tensor &Q, torch::Tensor &K, torch::Tensor &V,
                       torch::Tensor &O, const bool causal, const int64_t B,
                       const int64_t N, const int64_t H, const int64_t d) {
  // fp32 Flash Attention is only performant with
  // head dimension of 64. This is due to occupancy
  // problems caused by the larger amounts of SMEM used
  // by larger head dimensions
  static FMHADispatcher<1, 32, 1, 64, 64, 64> dispatcher;
  dispatcher.dispatch(Q.data_ptr<float>(), K.data_ptr<float>(),
                      V.data_ptr<float>(), O.data_ptr<float>(), causal,
                      static_cast<uint32_t>(B), static_cast<uint32_t>(N),
                      static_cast<int>(H), static_cast<int>(d));
}

torch::Tensor fmha_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                           bool causal) {
  TORCH_CHECK(Q.is_contiguous(), "Q must be row major");
  TORCH_CHECK(K.is_contiguous(), "K must be row major");
  TORCH_CHECK(V.is_contiguous(), "V must be row major");

  c10::ScalarType Q_dtype{Q.scalar_type()};
  c10::ScalarType K_dtype{K.scalar_type()};
  c10::ScalarType V_dtype{V.scalar_type()};

  TORCH_CHECK(Q_dtype == K_dtype && Q_dtype == V_dtype,
              "dtypes of all three projections must be the same");

  TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
              "shapes of all three projections must be the same");

  torch::Tensor O{torch::empty_like(Q)};

  const int64_t B{Q.size(0)}; // batch
  const int64_t N{Q.size(1)}; // sequence length
  const int64_t H{Q.size(2)}; // heads
  const int64_t d{Q.size(3)}; // head dim

  if (Q_dtype == torch::kFloat32) {
    fmha_forward_fp32(Q, K, V, O, causal, B, N, H, d);

    return O;
  } else {
    TORCH_CHECK(false, "currently only float32 FMHA is supported");
    return O;
  }
}

TORCH_LIBRARY(cobraml, m) {
  m.def("fmha(Tensor Q, Tensor K, Tensor V, bool causal) -> Tensor");
}

TORCH_LIBRARY_IMPL(cobraml, CUDA, m) { m.impl("fmha", &fmha_forward); }