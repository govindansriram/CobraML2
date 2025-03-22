cmake_minimum_required(VERSION 3.22)
project(cobraml VERSION 0.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

####################### libs #######################
find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)

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
        src/enums.cpp
        include/enums.h
        src/math_dis.cpp
        src/brarray.cpp
        include/brarray.h
)

add_library(cobraml SHARED ${SOURCES})

# Include directories for public headers
target_include_directories(cobraml PUBLIC include)

# Include directory for private headers
target_include_directories(cobraml PRIVATE src/standard_kernel)

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

################################################################

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cobraml PRIVATE ${TEST_COMPILE_OPTIONS})
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cobraml PRIVATE ${BENCH_COMPILE_OPTIONS})
endif ()

####################### test settings #######################
#add_executable(test_matrix tests/test_matrix.cpp)
add_executable(test_array tests/test_brarray.cpp)
#add_executable(test_tensor tests/test_tensor.cpp)
add_executable(test_enums tests/test_enums.cpp)

if (TEST_THREAD)
    list(APPEND TEST_COMPILE_OPTIONS -fsanitize=thread)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -g")
endif ()

if (TEST_ADDR)
    list(APPEND TEST_COMPILE_OPTIONS -fsanitize=address)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
endif ()

#target_compile_options(test_matrix PRIVATE ${TEST_COMPILE_OPTIONS})
target_compile_options(test_enums PRIVATE ${TEST_COMPILE_OPTIONS})
#target_compile_options(test_tensor PRIVATE ${TEST_COMPILE_OPTIONS})
target_compile_options(test_array PRIVATE ${TEST_COMPILE_OPTIONS})

#target_link_libraries(test_matrix gtest GTest::gtest_main cobraml)
target_link_libraries(test_array gtest GTest::gtest_main cobraml)
target_link_libraries(test_enums gtest GTest::gtest_main cobraml)
#target_link_libraries(test_tensor gtest GTest::gtest_main cobraml)

enable_testing()
include(GoogleTest)
gtest_discover_tests(test_enums test_array)

#########################################################


####################### Bench settings #######################
add_executable(benchmark_gemv benchmarks/benchmark_gemv.cpp)
target_compile_options(benchmark_gemv PRIVATE ${BENCH_COMPILE_OPTIONS})
target_link_libraries(benchmark_gemv benchmark::benchmark cobraml)
#########################################################
