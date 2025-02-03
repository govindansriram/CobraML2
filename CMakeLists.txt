cmake_minimum_required(VERSION 3.22)
project(ConBasedFilter VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(SOURCES
        include/matrix.h
        src/allocator.h
        src/math_dis.h
        src/standard_kernel/standard_allocator.h
        src/standard_kernel/standard_allocator.cpp
        src/allocator.cpp
        src/standard_kernel/standard_math.h
        src/standard_kernel/standard_math.cpp
        src/matrix.cpp
        src/enums.cpp
        include/enums.h
        src/math_dis.cpp
        src/barray.cpp
        include/barray.h
)

#list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/tbb/latest/lib/cmake") # path to intel onetbb

add_library(CmlContentBasedFiltering SHARED ${SOURCES})

# Include directories for public headers
target_include_directories(CmlContentBasedFiltering PUBLIC include)

target_include_directories(CmlContentBasedFiltering PRIVATE src/standard_kernel)

# Define install rules
install(TARGETS CmlContentBasedFiltering
        DESTINATION lib
)

install(DIRECTORY include/
        DESTINATION include
)

#find_package(TBB REQUIRED)
#target_link_libraries(CmlContentBasedFiltering PUBLIC TBB::tbb)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CmlContentBasedFiltering PUBLIC OpenMP::OpenMP_CXX)
endif()

## Specify the include directory where your header files are located

option(ENABLE_TESTING "Enable testing-specific compile options" ON)
option(IS_THREAD "Test for thread related issues" OFF)
option(VTUNE "Test for thread related issues" OFF)

if(DEFINED THREAD_COUNT)
    add_definitions(-DNUM_THREADS=${THREAD_COUNT})
endif()

if(ENABLE_TESTING)

    set(COMMON_COMPILE_OPTIONS
            -ggdb
            -pedantic-errors
            -O0
            -Wall
            -Weffc++
            -Wextra
            -Wconversion
            -Wsign-conversion
            -Werror
            -fopenmp
            -mavx2
    )

    target_compile_options(CmlContentBasedFiltering PRIVATE ${COMMON_COMPILE_OPTIONS})

    enable_testing()

    if (IS_THREAD)
        list(APPEND COMMON_COMPILE_OPTIONS -fsanitize=thread -fno-omit-frame-pointer -g)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -g")
    else()
        list(APPEND COMMON_COMPILE_OPTIONS -fsanitize=address)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g")
    endif ()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
    )

    # Make googletest available (this will download and build it)
    FetchContent_MakeAvailable(googletest)

    add_executable(test_matrix tests/test_matrix.cpp)
    add_executable(test_array tests/test_barray.cpp)
    add_executable(test_enums tests/test_enums.cpp)

    include(GoogleTest)
    gtest_discover_tests(test_matrix test_enums)

    target_compile_options(test_matrix PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_compile_options(test_enums PRIVATE ${COMMON_COMPILE_OPTIONS})

    target_link_libraries(
            test_matrix
            GTest::gtest_main
            CmlContentBasedFiltering
    )


    target_link_libraries(
            test_array
            GTest::gtest_main
            CmlContentBasedFiltering
    )

    target_link_libraries(
            test_enums
            GTest::gtest_main
            CmlContentBasedFiltering
    )

else ()

    find_package(benchmark REQUIRED)

    if(VTUNE)
        set(COMMON_COMPILE_OPTIONS
            -g
            -O2
            -mavx2
            -flto
            -fopenmp
        )
    else ()
        set(COMMON_COMPILE_OPTIONS
                -O2
                -flto
                -mavx2
                -fopenmp
        )
    endif ()

    # Append additional flags if provided via the command line
    if(DEFINED ADDITIONAL_COMPILE_OPTIONS)
        list(APPEND COMMON_COMPILE_OPTIONS ${ADDITIONAL_COMPILE_OPTIONS})
    endif()

    target_compile_options(CmlContentBasedFiltering PRIVATE ${COMMON_COMPILE_OPTIONS})

    add_definitions(-DBENCHMARK=1)

    message(STATUS "CMAKE_CXX_FLAGS: ${COMMON_COMPILE_OPTIONS}")

    add_executable(benchmark_matrix benchmarks/benchmark_matrix.cpp)

    target_compile_options(benchmark_matrix PRIVATE ${COMMON_COMPILE_OPTIONS})

    target_link_libraries(
            benchmark_matrix
            benchmark::benchmark
            CmlContentBasedFiltering
    )

endif()

