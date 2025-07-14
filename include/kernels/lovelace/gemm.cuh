//
// Created by sriram on 7/5/25.
//

#ifndef GEMM_CUH
#define GEMM_CUH
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "copy_builder.h"


namespace cobraml {
    using namespace cute;

    template<
        typename TypeA,
        typename TypeB,
        typename TypeC,
        typename ShapeA,
        typename ShapeB,
        typename ShapeC,
        typename StrideA,
        typename StrideB,
        typename StrideC
    >
    void gemm(
        const TypeA *a,
        const TypeB *b,
        TypeC *c,
        Layout<ShapeA, StrideA> layout_a,
        Layout<ShapeB, StrideB> layout_b,
        Layout<ShapeC, StrideC> layout_c
    ) {
        return;
    }

    template<
        typename ElementA,
        typename ElementB,
        typename SmemLayoutA,
        typename SmemLayoutB>
    struct SharedStorage {
        ArrayEngine<ElementA, cosize_v<SmemLayoutA> > A;
        ArrayEngine<ElementB, cosize_v<SmemLayoutB> > B;
    };

    template<
        typename CtaTiler,
        typename LayoutA,
        typename SmemLayoutA,
        typename TiledCopyA,
        typename LayoutB,
        typename SmemLayoutB,
        typename TiledCopyB,
        typename LayoutC,
        typename SmemLayoutC,
        typename TiledMMA
    >
    __global__ void gemm_device(
        CtaTiler cta_tiler,
        const half_t *a,
        const LayoutA layout_a,
        SmemLayoutA smem_layout_a,
        TiledCopyA copy_a,
        const half_t *b,
        const LayoutB layout_b,
        SmemLayoutB smem_layout_b,
        TiledCopyB copy_b,
        half_t *c,
        const LayoutC layout_c,
        SmemLayoutC smem_layout_c,
        TiledMMA mma
    ) {
        extern __shared__ char shared_memory[];

        Tensor global_tensor_a{make_tensor(make_gmem_ptr(a), layout_a)};                            // a global memory tensor for matrix A
        Tensor global_tensor_b{make_tensor(make_gmem_ptr(b), layout_b)};                            // a global memory tensor for matrix B
        Tensor global_tensor_c{make_tensor(make_gmem_ptr(c), layout_c)};                            // a global memory tensor for matrix C

        // here we get the coordinate of which matrix block we want to start at
        // for each global tensor notice blockIdx.x represents the x-axis (which is M),
        // blockIdx.y represents the y-axis which is N, _ is the z-axis which represents k,
        // this means that we want all blocks around the k dimension
        auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

        // since N is X we omit it, and here we say we want blockIdx.x for m and all K
        Tensor global_tile_a{local_tile(global_tensor_a, cta_tiler, cta_coord, Step<_1, X, _1>{})};    // Shape: (BLK_M, BLK_K, k)

        // since M is X we omit it, and here we say we want blockIdx.y for n and all K
        Tensor global_tile_b{local_tile(global_tensor_b, cta_tiler, cta_coord, Step<X, _1, _1>{})};    // Shape: (BLK_N, BLK_K, k)

        // since K is X we omit it, and here we say we want blockIdx.x for m and blockIdx.y for n
        Tensor global_tile_c{local_tile(global_tensor_c, cta_tiler, cta_coord, Step<_1, _1, X>{})};    // Shape: (BLK_M, BLK_N)

        using SharedStorage = SharedStorage<half_t, half_t, SmemLayoutA, SmemLayoutB>;
        SharedStorage& smem{*reinterpret_cast<SharedStorage*>(shared_memory)};
        Tensor shared_a{make_tensor(make_smem_ptr(smem.A.begin()), smem_layout_a)};                    // (BLK_M,BLK_K,PIPE)
        Tensor shared_b{make_tensor(make_smem_ptr(smem.B.begin()), smem_layout_b)};                    // (BLK_N,BLK_K,PIPE)

        // CopyA is the TiledCopyAtom, the get slice methods slices the
        // copy atom to encompass the copy responsibilities of a single thread,
        // that is what thr_copy_a does
        ThrCopy thr_copy_a{copy_a.get_slice(threadIdx.x)};

        // this is the source partition, we partition the source
        // Tensor which is the global tile a getting the tensor that
        // thread is responsible for

        // here's what the partition shape will look like
        // (CPY, CPY_Y, CPY_X, k)

        // CPY represents the amount of elements being copied this is typically
        // determined by the value layout

        // if the TiledCopy Layout is smaller than the tile_layout it will need
        // to be repeated in the x and y direction. That is what CPY_Y and CPY_X
        // represents

        // k will be the same value as the last dim in the parent tensor
        Tensor thread_part_glob_a{thr_copy_a.partition_S(global_tile_a)};                              // (CPY, CPY_M, CPY_K, k)
        Tensor thread_part_shared_a{thr_copy_a.partition_D(shared_a)};                                 // (CPY, CPY_M, CPY_K, PIPE)

        ThrCopy thr_copy_b{copy_b.get_slice(threadIdx.x)};
        Tensor thread_part_glob_b{thr_copy_b.partition_S(global_tile_b)};                              // (CPY, CPY_N, CPY_K, k)
        Tensor thread_part_shared_b{thr_copy_b.partition_D(shared_b)};                                 // (CPY, CPY_N, CPY_K, PIPE)

        CUTE_STATIC_ASSERT_V(size<1>(thread_part_glob_a) == size<1>(thread_part_shared_a));
        CUTE_STATIC_ASSERT_V(size<2>(thread_part_glob_a) == size<2>(thread_part_shared_a));

        CUTE_STATIC_ASSERT_V(size<1>(thread_part_glob_b) == size<1>(thread_part_shared_b));
        CUTE_STATIC_ASSERT_V(size<2>(thread_part_glob_b) == size<2>(thread_part_shared_b));

        auto K_PIPE_MAX{size<3>(thread_part_shared_a)};
        int k_tile_count{size<3>(thread_part_glob_a)};
        int k_tile_next{0};

        CUTE_UNROLL
        for (int k_pipe{0}; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
            copy(copy_a, thread_part_glob_a(_, _, _, k_tile_next), thread_part_shared_a(_, _, _, k_pipe));
            copy(copy_b, thread_part_glob_b(_, _, _, k_tile_next), thread_part_shared_b(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            // cant use breaks inside loop
            if (k_tile_count > 0){++k_tile_next;}
        }

        ThrMMA thr_mma{mma.get_slice(threadIdx.x)};

        // this partitions global tensor c based on the mma atom
        // it uses the same underlying buffer as the tensor
        // it will be of size (MMA,MMA_M,MMA_N)
        // MMA represents the shape of the data being held
        // MMA_M is how many times the MMA repeats in the M Mode
        // MMA_N is how many times the MMA repeats in the N Mode
        // This is determined  by the Atom Layout, the larger the
        // layout the fewer repetitions
        Tensor thread_part_view_global_c{thr_mma.partition_C(global_tile_c)};

        // fragments live in register and are allocated using the Array datatype
        // partition fragment creates a fragment with the specified partition shape,
        // since this is a fragment the underlying buffer is also an Array. The shape
        // will be the same but all the data will be contiguous
        Tensor thread_fragment_a{thr_mma.partition_fragment_A(shared_a(_, _, 0))};
        Tensor thread_fragment_b{thr_mma.partition_fragment_B(shared_b(_, _, 0))};

        // create a fragment the same size as the view
        Tensor thread_accum_c{thr_mma.make_fragment_C(thread_part_view_global_c)};

        clear(thread_accum_c);

        CUTE_STATIC_ASSERT_V(shape(thread_accum_c) == shape(thread_part_view_global_c));          // (MMA, MMA_M, MMA_N)
        CUTE_STATIC_ASSERT_V(size<1>(thread_part_view_global_c) == size<1>(thread_fragment_a));
        CUTE_STATIC_ASSERT_V(size<2>(thread_part_view_global_c) == size<1>(thread_fragment_b));



        if ((blockIdx.x + blockIdx.y == 0) && (threadIdx.x + threadIdx.y == 0)) {
            print(shared_a);
            printf("\n");
            print(shared_b);
            printf("\n");
            print(shape(thread_accum_c));
            printf("\n");
            print(shape(thread_fragment_a));
            printf("\n");
            print(shape(thread_fragment_b));
            // print(take<0, 3>(shape(thread_part_view_global_c)));
            // print(thr_mma.partition_A(shared_a(_, _, 0)));
            // printf("\n");
            // print(thread_fragment_a);
        }
    }

    template<
        typename ShapeA,
        typename ShapeB,
        typename ShapeC,
        typename StrideA,
        typename StrideB,
        typename StrideC
    >
    void runner_tn(
        const Brarray<half_t, ShapeA, StrideA> &a,
        const Brarray<half_t, ShapeB, StrideB> &b,
        Brarray<half_t, ShapeC, StrideC> &c) {
        auto prob_shape{
            make_shape(size<0>(ShapeC{}), size<1>(ShapeC{}), size<1>(ShapeA{}))
        };

        constexpr auto bM{_128{}};
        constexpr auto bN{_128{}};
        constexpr auto bK{_64{}};
        constexpr auto bP = _3{}; // Pipeline

        constexpr auto cta_tiler{make_shape(bM, bN, bK)};

        // Define the smem layouts (static)
        // Swizzles for LDSM and 128b k-major loads
        constexpr auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                                  Layout<Shape<_8, Shape<_8, _8> >,
                                                      Stride<_8, Stride<_1, _64> > >{});

        auto s_a = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
        auto s_b = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
        auto s_c = make_layout(make_shape(bM, bN));

        // make tiled_mma takes in 2 arguments
        constexpr TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
                                                 Layout<Shape<_2, _2> >{}, // 2x2x1 MMA Atoms
                                                 Tile<_32, _32, _16>{}); // 32x32x16 Tiled MMA for LDSM

        using CopyOp = AsyncTiledCopyBuilder<
            uint128_t,
            half_t,
            16,
            8,
            true,
            true>;

        auto copy_a{CopyOp::get_tiled_copy_A()};
        auto copy_b{CopyOp::get_tiled_copy_B()};

        int smem_size = int(sizeof(SharedStorage<half_t, half_t, decltype(s_a), decltype(s_b)>));

        const Copy_Atom<SM75_U32x4_LDSM_N, half_t> shared_to_register_atom_a;
        const Copy_Atom<SM75_U32x4_LDSM_N, half_t> shared_to_register_atom_b;

        dim3 dimBlock(size(mmaC));
        dim3 dimGrid(size(ceil_div(size<0>(prob_shape), bM)),
                     size(ceil_div(size<1>(prob_shape), bN)));

        // const half_t *  = a.get_ptr();

        auto kernel_fptr{
            gemm_device<
                decltype(cta_tiler),
                Layout<ShapeA, StrideA>,
                decltype(s_a),
                decltype(copy_a),
                Layout<ShapeB, StrideB>,
                decltype(s_b),
                decltype(copy_b),
                Layout<ShapeC, StrideC>,
                decltype(s_c),
                decltype(mmaC)>
        };

        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        kernel_fptr<<<dimGrid, dimBlock, smem_size>>>(
            cta_tiler,
            a.get_ptr(),
            Layout<ShapeA, StrideA>{},
            s_a,
            copy_a,
            b.get_ptr(),
            Layout<ShapeB, StrideB>{},
            s_b,
            copy_b,
            c.get_ptr(),
            Layout<ShapeC, StrideC>{},
            s_c,
            mmaC
        );

        CUTE_CHECK_LAST();

        // print(make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{}));
        // printf("\n");
        // print_latex(make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{}));
        // print_latex(make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{}));

        // print_latex(copyA);
        // print_latex(copyB);
    }
}

#endif //GEMM_CUH
