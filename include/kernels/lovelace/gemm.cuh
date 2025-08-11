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

    template<class ElementA,
        class ElementB,
        class SmemLayoutA,
        class SmemLayoutB>
    struct SharedStorage {
        cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA> > A;
        cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB> > B;
    };

    template<class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
        class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
        class TC, class CStride, class CSmemLayout, class TiledMma,
        class Alpha, class Beta>
    __global__ static
    __launch_bounds__(decltype(size(TiledMma{}))::value)
    void
    gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
                TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
                TC *C, CStride dC, CSmemLayout, TiledMma mma,
                Alpha alpha, Beta beta) {
        using namespace cute;

        // Preconditions
        CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
        CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

        CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
        CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // NumThreads

        static_assert(is_static<ASmemLayout>::value);
        static_assert(is_static<BSmemLayout>::value);
        static_assert(is_static<CSmemLayout>::value);

        CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
        CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
        CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
        CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
        CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
        CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

        CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA)); // dA strides for shape MK
        CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB)); // dB strides for shape NK
        CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC)); // dC strides for shape MN

        //
        // Full and Tiled Tensors
        //

        // Represent the full tensors
        Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
        Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
        Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)

        // Get the appropriate blocks for this thread block
        auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
        Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M,BLK_K,k)
        Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
        Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

        // Shared memory buffers
        extern __shared__ char shared_memory[];
        using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
        SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
        Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M,BLK_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N,BLK_K,PIPE)

        //
        // Partition the copying of A and B tiles across the threads
        //

        ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
        Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
        Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

        ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
        Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
        Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

        CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
        CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
        CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
        CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

        //
        // PREFETCH
        //

        auto K_PIPE_MAX = size<3>(tAsA);

        // Total count of tiles
        int k_tile_count = size<3>(tAgA);
        // Current tile index in gmem to read from
        int k_tile_next = 0;

        // Start async loads for all pipes but the last
        CUTE_UNROLL
        for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
            copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
            copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }

        //
        // Define A/B partitioning and C accumulators
        //

        ThrMMA thr_mma = mma.get_slice(threadIdx.x);
        Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

        // Allocate registers for pipelining
        Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
        Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
        // Allocate the accumulators -- same size as the projected data
        Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA,MMA_M,MMA_N)

        CUTE_STATIC_ASSERT_V(( shape(tCrC) == take<0,3>(shape(tCgC)))); // (MMA,MMA_M,MMA_N)
        CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA))); // MMA_M
        CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB))); // MMA_N

        // Clear the accumulators
        clear(tCrC);

        //
        // Copy Atom retiling
        //

        TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
        ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
        Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY,MMA_M,MMA_K,PIPE)
        Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA); // (CPY,MMA_M,MMA_K)

        TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
        ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
        Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY,MMA_N,MMA_K,PIPE)
        Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB); // (CPY,MMA_N,MMA_K)

#if 0
        if (thread0()) {
            print("  mA : ");
            print(mA);
            print("\n");
            print("  gA : ");
            print(gA);
            print("\n");
            print("  sA : ");
            print(sA);
            print("\n");
            print("tAgA : ");
            print(tAgA);
            print("\n");
            print("tAsA : ");
            print(tAsA);
            print("\n");
        }
#endif

#if 0
        if (thread0()) {
            print("  mB : ");
            print(mB);
            print("\n");
            print("  gB : ");
            print(gB);
            print("\n");
            print("  sB : ");
            print(sB);
            print("\n");
            print("tBgB : ");
            print(tBgB);
            print("\n");
            print("tBsB : ");
            print(tBsB);
            print("\n");
        }
#endif

#if 0
        if (thread0()) {
            print("  mC : ");
            print(mC);
            print("\n");
            print("  gC : ");
            print(gC);
            print("\n");
            print("tCgC : ");
            print(tCgC);
            print("\n");
            print("tCrA : ");
            print(tCrA);
            print("\n");
            print("tCrB : ");
            print(tCrB);
            print("\n");
            print("tCrC : ");
            print(tCrC);
            print("\n");

            print("tXsA : ");
            print(tXsA);
            print("\n");
            print("tXrA : ");
            print(tXrA);
            print("\n");
            print("tXsB : ");
            print(tXsB);
            print("\n");
            print("tXrB : ");
            print(tXrB);
            print("\n");
        }
#endif

#if 1

        // Current pipe index in smem to read from
        int smem_pipe_read = 0;
        // Current pipe index in smem to write to
        int smem_pipe_write = K_PIPE_MAX - 1;

        // Pipe slice
        Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
        Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

        // Size of the register pipeline
        auto K_BLOCK_MAX = size<2>(tCrA);
        CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

        // PREFETCH register pipeline
        if (K_BLOCK_MAX > 1) {
            // Wait until our first prefetched tile is loaded in
            cp_async_wait<K_PIPE_MAX - 2>();
            __syncthreads();

            // Prefetch the first rmem from the first k-tile
            copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
            copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
        }

        //
        // PIPELINED MAIN LOOP
        // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
        //           and explicit pipelines in shared memory.
        //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
        //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
        //   Data is computed on registers(b_block).
        //
        //   This allows all copies and compute to overlap:
        //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
        //     Copy from smem->rmem can overlap with compute on rmem.
        //

        CUTE_NO_UNROLL
        while (k_tile_count > -(K_PIPE_MAX - 1)) {
            CUTE_UNROLL
            for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
                if (k_block == K_BLOCK_MAX - 1) {
                    // Slice the smem_pipe_read smem
                    tXsA_p = tXsA(_, _, _, smem_pipe_read);
                    tXsB_p = tXsB(_, _, _, smem_pipe_read);

                    // Commit the smem for smem_pipe_read
                    cp_async_wait<K_PIPE_MAX - 2>();
                    __syncthreads();
                }

                // Load A, B shmem->regs for k_block+1
                auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX; // static
                copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
                copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
                // Copy gmem to smem before computing gemm on each k-pipe
                if (k_block == 0) {
                    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();

                    // Advance the gmem tile
                    --k_tile_count;
                    if (k_tile_count > 0) { ++k_tile_next; }

                    // Advance the smem pipe
                    smem_pipe_write = smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
                }
                // Thread-level register gemm for k_block
                gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
            }
        }

#endif

        //
        // Epilogue
        //

        axpby(alpha, tCrC, beta, tCgC);
    }

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
        typename TiledMMA,
        typename SharedToRegisterA,
        typename SharedToRegisterB
    >
    __global__ void gemm_device_2(
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
        TiledMMA mma,
        SharedToRegisterA shared_to_register_copy_atom_a,
        SharedToRegisterB shared_to_register_copy_atom_b
    ) {
        extern __shared__ char shared_memory[];

        Tensor global_tensor_a{make_tensor(make_gmem_ptr(a), layout_a)}; // a global memory tensor for matrix A
        Tensor global_tensor_b{make_tensor(make_gmem_ptr(b), layout_b)}; // a global memory tensor for matrix B
        Tensor global_tensor_c{make_tensor(make_gmem_ptr(c), layout_c)}; // a global memory tensor for matrix C

        // here we get the coordinate of which matrix block we want to start at
        // for each global tensor notice blockIdx.x represents the x-axis (which is M),
        // blockIdx.y represents the y-axis which is N, _ is the z-axis which represents k,
        // this means that we want all blocks around the k dimension
        auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

        // since N is X we omit it, and here we say we want blockIdx.x for m and all K
        Tensor global_tile_a{local_tile(global_tensor_a, cta_tiler, cta_coord, Step<_1, X, _1>{})};
        // Shape: (BLK_M, BLK_K, k)

        // since M is X we omit it, and here we say we want blockIdx.y for n and all K
        Tensor global_tile_b{local_tile(global_tensor_b, cta_tiler, cta_coord, Step<X, _1, _1>{})};
        // Shape: (BLK_N, BLK_K, k)

        // since K is X we omit it, and here we say we want blockIdx.x for m and blockIdx.y for n
        Tensor global_tile_c{local_tile(global_tensor_c, cta_tiler, cta_coord, Step<_1, _1, X>{})};
        // Shape: (BLK_M, BLK_N)

        using SharedStorage = SharedStorage<half_t, half_t, SmemLayoutA, SmemLayoutB>;
        SharedStorage &smem{*reinterpret_cast<SharedStorage *>(shared_memory)};
        Tensor shared_a{make_tensor(make_smem_ptr(smem.A.begin()), smem_layout_a)}; // (BLK_M,BLK_K,PIPE)
        Tensor shared_b{make_tensor(make_smem_ptr(smem.B.begin()), smem_layout_b)}; // (BLK_N,BLK_K,PIPE)

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
        Tensor thread_part_glob_a{thr_copy_a.partition_S(global_tile_a)}; // (CPY, CPY_M, CPY_K, k)
        Tensor thread_part_shared_a{thr_copy_a.partition_D(shared_a)}; // (CPY, CPY_M, CPY_K, PIPE)

        ThrCopy thr_copy_b{copy_b.get_slice(threadIdx.x)};
        Tensor thread_part_glob_b{thr_copy_b.partition_S(global_tile_b)}; // (CPY, CPY_N, CPY_K, k)
        Tensor thread_part_shared_b{thr_copy_b.partition_D(shared_b)}; // (CPY, CPY_N, CPY_K, PIPE)

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
            if (k_tile_count > 0) { ++k_tile_next; }
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

        // all partitions of fragments are dominantly determined by the atom layout not the
        // permutations

        clear(thread_accum_c);

        CUTE_STATIC_ASSERT_V(shape(thread_accum_c) == shape(thread_part_view_global_c)); // (MMA, MMA_M, MMA_N)
        CUTE_STATIC_ASSERT_V(size<1>(thread_part_view_global_c) == size<1>(thread_fragment_a));
        CUTE_STATIC_ASSERT_V(size<2>(thread_part_view_global_c) == size<1>(thread_fragment_b));

        TiledCopy shared_2_register_tiled_copy_a{make_tiled_copy_A(shared_to_register_copy_atom_a, mma)};
        TiledCopy shared_2_register_tiled_copy_b{make_tiled_copy_B(shared_to_register_copy_atom_b, mma)};

        ThrCopy shared_2_register_thread_copy_a{shared_2_register_tiled_copy_a.get_slice(threadIdx.x)};

        Tensor shared_part_for_reg_a{shared_2_register_thread_copy_a.partition_S(shared_a)};

        if (thread0()) {
            // print_latex(shared_2_register_tiled_copy_b);
            // print(thread_fragment_a);
            // printf("\n");
            // print(shared_part_for_reg_a);
            // printf("\n");
            // print(shape(thread_accum_c));
            // printf("\n");
            // print(shape(thread_fragment_a));
            // printf("\n");
        }
    }

    template<class Alpha, class Beta>
    void
    runner_tn(
        const int m,
        const int n,
        const int k,
        Alpha alpha,
        half_t const *A,
        int ldA,
        half_t const *B,
        int ldB,
        Beta beta,
        half_t *C,
        int ldC,
        cudaStream_t stream = 0) {
        using namespace cute;

        // Define shapes (dynamic)
        auto M = int(m);
        auto N = int(n);
        auto K = int(k);
        auto prob_shape = make_shape(M, N, K); // (M, N, K)

        // Define TN strides (mixed)
        auto dA = make_stride(ldA, Int<1>{}); // (dM, dK)
        auto dB = make_stride(ldB, Int<1>{}); // (dN, dK)
        auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

        // Define CTA tile sizes (static)
        auto bM = Int<128>{};
        auto bN = Int<128>{};
        auto bK = Int<64>{};
        auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
        auto bP = Int<3>{}; // Pipeline

        // Define the smem layouts (static)
        // Swizzles for LDSM and 128b k-major loads
        auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                        Layout<Shape<_8, Shape<_8, _8> >,
                                            Stride<_8, Stride<_1, _64> > >{});

        auto sA = tile_to_shape(swizzle_atom, make_shape(bM, bK, bP));
        auto sB = tile_to_shape(swizzle_atom, make_shape(bN, bK, bP));
        auto sC = make_layout(make_shape(bM, bN));

        // Define the thread layouts (static)

        TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                          Layout<Shape<_16, _8>, Stride<_8, _1> >{}, // Thr layout 16x8 k-major
                                          Layout<Shape<_1, _8> >{}); // Val layout  1x8 k-major
        TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                          Layout<Shape<_16, _8>, Stride<_8, _1> >{}, // Thr layout 16x8 k-major
                                          Layout<Shape<_1, _8> >{}); // Val layout  1x8 n-major

        TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                       Layout<Shape<_2, _2> >{}, // 2x2x1 MMA Atoms
                                       Tile<_32, _32, _16>{}); // 32x32x16 Tiled MMA for LDSM

        //Copy_Atom<DefaultCopy, half_t> s2r_atom_A;
        //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_A;
        //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_A;
        //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_A;
        Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;

        //Copy_Atom<DefaultCopy, half_t> s2r_atom_B;
        //Copy_Atom<UniversalCopy<half_t>, half_t> s2r_atom_B;
        //Copy_Atom<SM75_U32x1_LDSM_N, half_t> s2r_atom_B;
        //Copy_Atom<SM75_U32x2_LDSM_N, half_t> s2r_atom_B;
        Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

#if 0
        print(copyA);
        print(copyB);
        print(mmaC);
#endif

#if 0
        print_latex(copyA);
        print_latex(copyB);
        print_latex(mmaC);
#endif

        int smem_size = int(sizeof(SharedStorage<cute::half_t, cute::half_t, decltype(sA), decltype(sB)>));
        dim3 dimBlock(size(mmaC));
        dim3 dimGrid(size(ceil_div(M, bM)),
                     size(ceil_div(N, bN)));

        auto kernel_fptr = gemm_device<
            decltype(prob_shape), decltype(cta_tiler),
            cute::half_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
            cute::half_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
            cute::half_t, decltype(dC), decltype(sC), decltype(mmaC),
            decltype(alpha), decltype(beta)>;

        // Set L1 to be SMEM only
        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);

        kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
        (prob_shape, cta_tiler,
         A, dA, sA, copyA, s2r_atom_A,
         B, dB, sB, copyB, s2r_atom_B,
         C, dC, sC, mmaC,
         alpha, beta);
    }

    template<
        typename ShapeA,
        typename ShapeB,
        typename ShapeC,
        typename StrideA,
        typename StrideB,
        typename StrideC
    >
    void gemm_tn(
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

        TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                          Layout<Shape<_16, _8>, Stride<_8, _1> >{}, // Thr layout 16x8 k-major
                                          Layout<Shape<_1, _8> >{});

        // print_latex(make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
        //                      Layout<Shape<_2,_2>>{},
        //                      Tile<_32,_16,_8>{})); // 32x32x16 Tiled MMA for LDSM);

        printf("%d, %d, %d", dimBlock.x, dimBlock.y, dimBlock.z);

        // print_layout(Layout<Shape<_8, Shape<_8, _8> >,
        //     Stride<_8, Stride<_1, _64> > >{});
        //
        // printf("\n");
        // print_layout(swizzle_atom);

        // auto zipped_layout = zipped_divide(s_a, make_shape(bM, bK, _1{}));
        // const half_t *  = a.get_ptr();

        // printf("\n");
        // print(select<0, 1>(layout<0>(zipped_layout)));

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
                decltype(mmaC),
                decltype(shared_to_register_atom_a),
                decltype(shared_to_register_atom_b)
            >
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
            mmaC,
            shared_to_register_atom_a,
            shared_to_register_atom_b
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
