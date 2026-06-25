#include <cobraml2/runtime/kv_cache.cuh>
#include <gtest/gtest.h>

using namespace cobraml::runtime;

TEST(KV_CACHE, MHA_KV_CACHE_GET) {
  const int num_pages{10};

  KVCacheMHA<float, 4, 32, 256, 16> cache(num_pages);
  KVManager manager(cache);

  for (int i{0}; i < num_pages; ++i) {
    int index{manager.get_page()};
    ASSERT_EQ(num_pages - 1 - i, index);
  }

  ASSERT_ANY_THROW(manager.get_page());
}

TEST(KV_CACHE, MHA_KV_CACHE_FREE) {
  const int num_pages{10};

  KVCacheMHA<float, 4, 32, 256, 16> cache(num_pages);
  KVManager manager(cache);

  ASSERT_ANY_THROW(manager.free_page(11));
  ASSERT_ANY_THROW(manager.free_page(-1));
  ASSERT_ANY_THROW(manager.free_page(0));

  for (int i{0}; i < num_pages; ++i)
    manager.get_page();

  ASSERT_ANY_THROW(manager.free_page(-1));

  for (int i{0}; i < num_pages; ++i)
    manager.free_page(i);

  ASSERT_ANY_THROW(manager.free_page(0));
}