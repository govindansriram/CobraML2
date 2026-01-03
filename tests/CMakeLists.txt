include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
FetchContent_MakeAvailable(googletest)

add_library(test_common INTERFACE)
target_include_directories(test_common INTERFACE 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add test executables
add_executable(test_fmha_cc test_fmha_cc.cu)
target_link_libraries(test_fmha_cc test_common cobraml2 GTest::gtest_main CUDA::curand)
add_test(NAME test_fmha_cc COMMAND test_fmha_cc)

add_executable(test_mha test_mha.cu)
target_link_libraries(test_mha test_common cobraml2 GTest::gtest_main CUDA::curand)
add_test(NAME test_mha COMMAND test_mha)