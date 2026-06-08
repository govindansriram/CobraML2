#pragma once
#include <concepts>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <iostream>
#include <thrust/device_vector.h>
#include <utility>

namespace cobraml::runtime {

using namespace cute;

template <typename C>
concept KVCache = requires(C cache, int page) {
  typename C::PageTensorType;
  { cache.get_page(page) } -> std::same_as<typename C::PageTensorType>;
  { cache.num_pages() } -> std::convertible_to<int>;
};

template <typename TensorDType, int num_layers, int num_heads, int head_dim,
          int page_size>
struct KVCacheMHA {

  using NumLayersType = Int<num_layers>;
  using NumHeadsType = Int<num_heads>;
  using HeadDimType = Int<head_dim>;
  using PageSizeType = Int<page_size>;

  static constexpr int elements_per_kv_page{2 * num_layers * num_heads *
                                            head_dim * page_size};

  using CacheLayoutType =
      decltype(make_layout(make_shape(16, _2{}, NumLayersType{}, NumHeadsType{},
                                      PageSizeType{}, HeadDimType{}),
                           LayoutRight{}));

  using TensorType = decltype(make_tensor(
      make_gmem_ptr(static_cast<TensorDType *>(nullptr)), CacheLayoutType{}));

  using PageTensorType = decltype(std::declval<TensorType>()(0, _, _, _, _, _));

  thrust::device_vector<TensorDType> buffer;
  TensorType buffer_tensor;
  const int n_pages;

  static TensorType create_tensor(const int n_pages, TensorDType *buffer_ptr) {
    const CacheLayoutType layout{
        make_layout(make_shape(n_pages, _2{}, NumLayersType{}, NumHeadsType{},
                               PageSizeType{}, HeadDimType{}),
                    LayoutRight{})};

    return make_tensor(make_gmem_ptr(buffer_ptr), layout);
  }

  KVCacheMHA(int pages)
      : buffer(elements_per_kv_page * pages),
        buffer_tensor(
            create_tensor(pages, thrust::raw_pointer_cast(buffer.data()))),
        n_pages(pages) {}

  TensorType get_cache() const { return buffer_tensor; }

  PageTensorType get_page(const int page) const {
    return buffer_tensor(page, _, _, _, _, _);
  }

  int num_pages() const { return n_pages; }
};

static_assert(KVCache<KVCacheMHA<float, 12, 12, 64, 16>>);

template <KVCache CacheType> class KVManager {

  const CacheType &cache;
  std::vector<bool> page_map;
  std::vector<int> available_pages;

  using PageTensorType = typename CacheType::PageTensorType;

public:
  explicit KVManager(const CacheType &cache)
      : cache(cache), page_map(cache.num_pages(), true),
        available_pages(cache.num_pages(), 0) {
    for (int i{0}; i < cache.num_pages(); ++i)
      available_pages[i] = i;
  }

  cute::tuple<int, PageTensorType> get_page() {
    if (available_pages.empty())
      throw std::out_of_range("no more available pages");

    const int page{available_pages.back()};
    available_pages.pop_back();
    page_map[page] = false;
    return make_tuple(page, cache.get_page(page));
  }

  void free_page(int page) {
    if (page < 0 || page >= cache.num_pages())
      throw std::out_of_range("invalid page was provided");
    if (page_map[page])
      throw std::logic_error("attempting to free a free page");
    page_map[page] = true;
    available_pages.push_back(page);
  }
};
} // namespace cobraml::runtime