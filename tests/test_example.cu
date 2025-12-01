#include <gtest/gtest.h>
#include <concepts>
#include <span>
#include <ranges>
#include <array>
#include <cute/layout.hpp>

// Concept
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

__global__ void kernel(){
    cute::print(threadIdx.x);
}

// Consteval (must run at compile time)
consteval int square(int n) {
    return n * n;
}

// Function using concept
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

TEST(Cpp20Test, Concepts) {
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(1.5, 2.5), 4.0);
}

TEST(Cpp20Test, Consteval) {
    constexpr int result = square(5);
    EXPECT_EQ(result, 25);
}

TEST(Cpp20Test, Span) {
    std::array<int, 4> arr = {1, 2, 3, 4};
    std::span<int> s(arr);
    EXPECT_EQ(s.size(), 4);
    EXPECT_EQ(s[0], 1);
}

TEST(Cpp20Test, Ranges) {
    std::array<int, 4> arr = {1, 2, 3, 4};
    auto doubled = arr | std::views::transform([](int x) { return x * 2; });
    kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    EXPECT_EQ(*doubled.begin(), 2);
}

TEST(Cpp20Test, DesignatedInitializers) {
    struct Point { int x; int y; };
    Point p = {.x = 10, .y = 20};
    EXPECT_EQ(p.x, 10);
    EXPECT_EQ(p.y, 20);
}