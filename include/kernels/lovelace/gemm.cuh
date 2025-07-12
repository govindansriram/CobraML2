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
        Tensor global_tile_x{local_tile(global_tensor_c, cta_tiler, cta_coord, Step<_1, _1, X>{})};    // Shape: (BLK_M, BLK_N)

        using SharedStorage = SharedStorage<half_t, half_t, SmemLayoutA, SmemLayoutB>;
        SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
        Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), smem_layout_a);                         // (BLK_M,BLK_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), smem_layout_b);                         // (BLK_N,BLK_K,PIPE)

        // CopyA is the TiledCopyAtom, the get slice methods slices the
        // copy atom to encompass the copy responsibilities of a single thread,
        // that is what thr_copy_a does
        ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);

        // this is the source partition, we partition the source
        // Tensor which is hte global tile a getting the tensor that
        // thread is responsible for

        // here's what the partition shape will look like
        // k will be the same value as k in the parent tensor
        // being partitioned as this represents the actual
        // number of repetitions

        // the Tiled Copy Layout may be smaller than the block size
        // so CPY represents how many times the copy has to repeat in
        // the x and y direction
        Tensor tAgA = thr_copy_a.partition_S(global_tile_a);                                           // (CPY,CPY_M,CPY_K,k)
        Tensor tAsA = thr_copy_a.partition_D(sA);                                                      // (CPY,CPY_M,CPY_K,PIPE)



        if ((blockIdx.x + blockIdx.y == 0) && (threadIdx.x + threadIdx.y == 0)) {
            print(global_tensor_a);
            printf("\n");
            print(sA);
            printf("\n");
            print(global_tile_a);
            printf("\n");
            print(tAgA);
            printf("\n");
            print(tAsA);
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
        constexpr auto bP = Int<3>{}; // Pipeline

        constexpr auto cta_tiler{make_shape(bM, bN, bK)};

        // Define the smem layouts (static)
        // Swizzles for LDSM and 128b k-major loads
        constexpr auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                                  Layout<Shape<_8, Shape<_8, _8> >,
                                                      Stride<_8, Stride<_1, _64> > >{});

        auto s_a = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
        auto s_b = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
        auto s_c = make_layout(make_shape(bM, bN));


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

        // print_latex(copyA);
        // print_latex(copyB);
    }
}

#endif //GEMM_CUH
