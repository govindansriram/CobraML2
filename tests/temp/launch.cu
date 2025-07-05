//
// Created by sriram on 6/28/25.
//

#include "layout_add.cuh"

void run_func() {
    using namespace cute;

    Layout<Shape <_8,_4>, Stride<_1,_4>> A_layout{};
    Layout<Shape <_8,_4>, Stride<_1,_4>> B_layout{};

    int a = 10;
    int b = 10;

    tensor_add<<<1, 1>>>(&a, &b, A_layout, B_layout);

    cudaDeviceSynchronize();


}