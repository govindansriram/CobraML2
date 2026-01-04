cmake_minimum_required(VERSION 3.24)
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(cobraml2 VERSION 0.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

option(BENCHMARK "Enable benchmarking" OFF)
if(BENCHMARK)
    add_compile_definitions(BENCHMARK)
endif()

# Performance boost
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# attaches lineinfo for profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# add's linker details, shared memory usage and register spills
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills")

# Full template error messages, Allow type punning, Silence ABI warnings, Pass unknown flags to gcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep")

add_library(cobraml2 INTERFACE)
target_include_directories(cobraml2 INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(cobraml2 INTERFACE external/cutlass/include)
target_include_directories(cobraml2 INTERFACE external/cutlass/tools/util/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cobraml2 INTERFACE CUDA::toolkit)

option(COBRAML2_BUILD_TESTS "Build tests" ON)
if(COBRAML2_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()