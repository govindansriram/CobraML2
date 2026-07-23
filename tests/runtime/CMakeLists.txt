add_executable(test_kv_cache test_kv_cache.cu)
target_link_libraries(test_kv_cache test_common cobraml2 GTest::gtest_main)
add_test(NAME test_kv_cache COMMAND test_kv_cache)
