add_executable(test_fmha_cc test_fmha_cc.cu)
target_link_libraries(test_fmha_cc test_common cobraml2 GTest::gtest_main CUDA::curand)
add_test(NAME test_fmha_cc COMMAND test_fmha_cc)

add_executable(test_paged_fmha_cc test_paged_fmha_cc.cu)
target_link_libraries(test_paged_fmha_cc test_common cobraml2 GTest::gtest_main CUDA::curand)
add_test(NAME test_paged_fmha_cc COMMAND test_paged_fmha_cc)
