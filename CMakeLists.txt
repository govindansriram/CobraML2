cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_ARCHITECTURES 89) # reduce build times to incorporate more archs add 52 60 75
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(cobraml VERSION 0.0.0 LANGUAGES CXX CUDA)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

####################### libs #######################
find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

# releases can be found here https://github.com/google/googletest/releases always pick the link with zip
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)

FetchContent_MakeAvailable(googletest)
####################################################

set(SOURCES
        src/allocator.h
        src/math_dis.h
        src/standard_kernel/standard_allocator.h
        src/standard_kernel/standard_allocator.cpp
        src/allocator.cpp
        src/standard_kernel/standard_math.h
        src/standard_kernel/standard_math.cpp
        src/standard_kernel/gemm.cpp
        src/standard_kernel/eq.cpp
        src/enums.cpp
        include/enums.h
        src/math_dis.cpp
        src/brarray.cpp
        include/brarray.h
        src/autograd/computation_graph.cpp
        src/autograd/computation_graph.h
        src/autograd/layers/element_wise.cpp
        src/autograd/layers/element_wise.h
        src/cuda_standard_kernel/cuda_allocator.cpp
        src/cuda_standard_kernel/cuda_allocator.h
        src/cuda_standard_kernel/cuda_helpers.h
        src/instantiate.h
        src/cuda_standard_kernel/gemv.cu
        src/cuda_standard_kernel/cuda_math.h
        src/cuda_standard_kernel/cuda_math.cpp
)

set(CUDA_SOURCES
        src/cuda_standard_kernel/cuda_device_helpers.cuh
        src/cuda_standard_kernel/gemv.cu
        src/cuda_standard_kernel/gemm.cu
        src/cuda_standard_kernel/eq.cu
)

add_library(cobraml SHARED ${SOURCES})
target_sources(cobraml PRIVATE ${CUDA_SOURCES})

# Include directories for public headers
target_include_directories(cobraml PUBLIC include)

# Include directory for private headers
target_include_directories(cobraml PRIVATE src/standard_kernel src/autograd src/autograd/layers src/cuda_standard_kernel CUDA::toolkit)
target_link_libraries(cobraml PRIVATE CUDA::cudart)

set_target_properties(cobraml PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

# Define install rules
install(TARGETS cobraml
        DESTINATION lib
)

install(DIRECTORY include/
        DESTINATION include
)

target_link_libraries(cobraml PUBLIC OpenMP::OpenMP_CXX)
####################################################


####################### compiler settings #######################

if (DEFINED THREAD_COUNT)
    add_definitions(-DNUM_THREADS=${THREAD_COUNT})
endif ()

set(TEST_COMPILE_OPTIONS
        -pedantic-errors
        -fno-omit-frame-pointer
        -O0
        -Wall
        -Weffc++
        -Wextra
        -Wconversion
        -Wsign-conversion
        -Werror
        -fopenmp
)

set(BENCH_COMPILE_OPTIONS
        -O3
        -fopenmp
        -DBENCHMARK=1
)

option(AVX2 "use AVX2" OFF)
option(PROFILER "add -g flag to help profilers" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND TEST_COMPILE_OPTIONS -ggdb)
else ()
    list(APPEND TEST_COMPILE_OPTIONS -g)
endif ()

if (AVX2)
    list(APPEND TEST_COMPILE_OPTIONS -march=native -mavx2 -mfma)  # Add AI-related file to the source list
    list(APPEND BENCH_COMPILE_OPTIONS -march=native -mavx2 -mfma)  # Add AI-related file to the source list
    add_definitions(-DAVX2)
endif ()

if (PROFILER)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND BENCH_COMPILE_OPTIONS -ggdb)
    else ()
        list(APPEND BENCH_COMPILE_OPTIONS -g)
    endif ()
endif ()

############################### ADD cuda compilation flags ###############################################
set(CUDA_COMPILE_OPTIONS_DEBUG
        -O0
        -G
        --ptxas-options=-v
)

set(CUDA_COMPILE_OPTIONS_RELEASE
        -O3
        -DBENCHMARK=1
)

# Apply different flags based on build type
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cobraml PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${TEST_COMPILE_OPTIONS}>  # Apply to C++ files only
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS_DEBUG}>  # Apply to CUDA files
    )
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cobraml PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${BENCH_COMPILE_OPTIONS}>  # Apply to C++ files only
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS_RELEASE}>  # Apply to CUDA files
    )
endif ()

####################### test settings #######################
add_executable(test_cpu_math tests/test_cpu_math.cpp)
add_executable(test_array tests/test_brarray.cpp)
add_executable(test_enums tests/test_enums.cpp)
add_executable(test_array_gpu tests/test_brarray_gpu.cpp)

target_include_directories(test_array PRIVATE ${PROJECT_SOURCE_DIR}/include)

if (TEST_THREAD)
    list(APPEND TEST_COMPILE_OPTIONS -fsanitize=thread)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -g")
endif ()

if (TEST_ADDR)
    list(APPEND TEST_COMPILE_OPTIONS -fsanitize=address)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
endif ()

# CPU
target_compile_options(test_enums PRIVATE ${TEST_COMPILE_OPTIONS})
target_compile_options(test_array PRIVATE ${TEST_COMPILE_OPTIONS})
target_compile_options(test_cpu_math PRIVATE ${TEST_COMPILE_OPTIONS})

## GPU
#target_compile_options(test_array_gpu PRIVATE ${TEST_COMPILE_OPTIONS})
#target_compile_options(test_array_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS_DEBUG}>)

target_link_libraries(test_array gtest GTest::gtest_main cobraml)
target_link_libraries(test_enums gtest GTest::gtest_main cobraml)
target_link_libraries(test_cpu_math gtest GTest::gtest_main cobraml)
target_link_libraries(test_array_gpu gtest GTest::gtest_main cobraml)


enable_testing()
include(GoogleTest)
gtest_discover_tests(test_enums test_array, test_cpu_math, test_array_gpu)
#########################################################


####################### Bench settings #######################
add_executable(benchmark_gemv benchmarks/benchmark_gemv.cpp)
target_compile_options(benchmark_gemv PRIVATE ${BENCH_COMPILE_OPTIONS})
target_link_libraries(benchmark_gemv benchmark::benchmark cobraml)

add_executable(benchmark_eq benchmarks/eq_cuda.cpp)
target_compile_options(benchmark_eq PRIVATE ${BENCH_COMPILE_OPTIONS})
target_link_libraries(benchmark_eq benchmark::benchmark cobraml)

add_executable(benchmark_cuda_matrix benchmarks/gemm_cuda.cpp)
target_compile_options(benchmark_cuda_matrix PRIVATE ${BENCH_COMPILE_OPTIONS})
target_link_libraries(benchmark_cuda_matrix benchmark::benchmark cobraml)

add_executable(benchmark_cuda_gemv benchmarks/gemv_cuda.cpp)
target_compile_options(benchmark_cuda_gemv PRIVATE ${BENCH_COMPILE_OPTIONS})
target_link_libraries(benchmark_cuda_gemv benchmark::benchmark cobraml)
#########################################################
