cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_ARCHITECTURES "89;89-virtual") # reduce build times to incorporate more archs add 52 60 75
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(cobraml VERSION 0.0.0 LANGUAGES CXX CUDA)

####################### libs #######################
find_package(benchmark REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(/home/sriram/Applications/cutlass/cutlass/include)
include_directories(/home/sriram/Applications/cutlass/cutlass/tools/util/include)
include_directories(include)
include_directories(CUDA::toolkit)

# releases can be found here https://github.com/google/googletest/releases always pick the link with zip
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)

FetchContent_MakeAvailable(googletest)
####################################################

add_executable(
        test_brarray
        tests/test_brarray.cu
)

add_executable(
        benchmark_ampere
        benchmarks/benchmark_ampere.cu
)

function(setup_cuda_target target_name)
    cmake_parse_arguments(
            ARG                        # Prefix for parsed args (can be empty)
            "WITH_GTEST"               # Optional single-value flags
            ""                         # Optional single-value args (none)
            ""                         # Optional multi-value args (none)
            ${ARGN}                    # Remaining arguments passed to the function
    )

    set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    if(ARG_WITH_GTEST)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cublas gtest GTest::gtest_main)
    else()
        target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cublas benchmark::benchmark)
    endif()

    target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --keep>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=compute_89>
    )
endfunction()

setup_cuda_target(test_brarray WITH_GTEST)
setup_cuda_target(benchmark_ampere)
