include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
FetchContent_MakeAvailable(googletest)

# Add test executables
add_executable(test_example test_example.cu)
target_link_libraries(test_example cobraml2 GTest::gtest_main)
add_test(NAME test_example COMMAND test_example)