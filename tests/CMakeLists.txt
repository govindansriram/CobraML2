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

add_subdirectory(kernels)
add_subdirectory(runtime)
