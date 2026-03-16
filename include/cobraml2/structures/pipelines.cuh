#pragma once
#include <concepts>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "../macros.cuh"
#include "./load.cuh"


namespace cobraml::pipelines{
using namespace cute;

template<typename V>
concept PipelineViewType = requires {
    typename V::ProducerBarrierArrayType;
    typename V::ConsumerBarrierArrayType;
    typename V::TileTensorTypeA;
    typename V::TileTensorTypeB;
    { V::arrival_count } -> std::convertible_to<size_t>;
    { V::static_pipeline_stages } -> std::convertible_to<size_t>;
} && std::constructible_from<
    V,
    typename V::TileTensorTypeA&,
    typename V::TileTensorTypeB&,
    typename V::ProducerBarrierArrayType&,
    typename V::ConsumerBarrierArrayType&
>;

template<typename T>
concept ThreadRoleCompatible = std::equality_comparable<T>;

struct State {
    int index{0};
    uint32_t phase{0};
    size_t pipeline_stages{0};

    COBRA_DEVICE State& operator++() {
        ++index;
        if (index >= pipeline_stages) {
            index = 0;
            phase ^= 1;
        }

        return *this;
    }
};

enum class BarrierStatus : uint32_t {
    WaitDone = 1,
    WaitAgain = 0
};

enum class ThreadRole {
    Load,
    MMA
};

namespace sm100{
template<
    typename TileLayoutTypeA,
    typename TileLayoutTypeB,
    typename AType,
    typename BType,
    size_t pipeline_stages
>
struct TMAProducerView {

    using LoadTypeA = loadop::sm100::TMALoad<AType, true>;
    using LoadTypeB = loadop::sm100::TMALoad<BType, false>;

    using ProducerBarrierType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarrierType = cutlass::arch::ClusterBarrier;

    using ProducerBarrierArrayType = ProducerBarrierType[pipeline_stages];
    using ConsumerBarrierArrayType = ConsumerBarrierType[pipeline_stages];

    ProducerBarrierArrayType &producer_barrier;
    ConsumerBarrierArrayType &consumer_barrier;

    using TileTensorTypeA = Tensor<ViewEngine<smem_ptr<AType*>>, TileLayoutTypeA>;
    using TileTensorTypeB = Tensor<ViewEngine<smem_ptr<BType*>>, TileLayoutTypeB>;

    TileTensorTypeA &a_tensor;
    TileTensorTypeB &b_Tensor;

    bool is_leader{false};

    private:
    static constexpr size_t transaction_bytes{
        (cosize_v<TileLayoutTypeA> * sizeof(AType) + cosize_v<TileLayoutTypeB> * sizeof(BType)) / pipeline_stages
    };

    public:
    COBRA_DEVICE TMAProducerView(
        TileTensorTypeA& a_smem_tensor,
        TileTensorTypeB& b_smem_tensor,
        ProducerBarrierArrayType& producer_barrier,
        ConsumerBarrierArrayType& consumer_barrier
    ):
        a_tensor(a_smem_tensor), b_Tensor(b_smem_tensor),
        producer_barrier(producer_barrier), consumer_barrier(consumer_barrier) {
        is_leader = (cute::elect_one_sync() == 1);
    }

    static constexpr size_t arrival_count{1};
    static constexpr size_t static_pipeline_stages{pipeline_stages};

    /**
     * @brief preemptively checks if the barrier needs to be waited on
     * ready to be filled
     * @param state
     * @return BarrierStatus
     */
    COBRA_DEVICE BarrierStatus check_barrier(State& state){
        return static_cast<BarrierStatus>(consumer_barrier[state.index].try_wait(state.phase));
    }

    /**
     * @brief gets the shared memory tiles once they are available 
     * 
     * @param state encapsulates the stage in the pipeline we are on
     * @param status whether to skip the wait or not
     * @return a tuple of the a and b slice 
     */
    COBRA_DEVICE auto get_tiles(State& state, BarrierStatus status){
        if (status == BarrierStatus::WaitAgain) 
            consumer_barrier[state.index].wait(state.phase);

        // we set both arrive and expect bytes because: 
        // if we had just arrive then we would have to wait for the 
        // pipeline to finish before arriving, if we had just expect bytes
        // the pipeline would instantly arrive at the start since expect bytes
        // starts at 0
        if (is_leader)
            producer_barrier[state.index].arrive_and_expect_tx(transaction_bytes);

        return make_tuple(
            a_tensor(_, _, _, state.index), 
            b_Tensor(_, _, _, state.index)
        );
    }

    COBRA_S_DEVICE State initial_state() {
        return State{0, 1};
    }
};


template<
    typename TileLayoutTypeA,
    typename TileLayoutTypeB,
    typename AType,
    typename BType,
    size_t pipeline_stages
>
struct UMMAConsumerView {

    using ProducerBarrierType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarrierType = cutlass::arch::ClusterBarrier;

    using ProducerBarrierArrayType = ProducerBarrierType[pipeline_stages];
    using ConsumerBarrierArrayType = ConsumerBarrierType[pipeline_stages];

    ProducerBarrierArrayType &producer_barrier;
    ConsumerBarrierArrayType &consumer_barrier;

    using TileTensorTypeA = Tensor<ViewEngine<smem_ptr<AType*>>, TileLayoutTypeA>;
    using TileTensorTypeB = Tensor<ViewEngine<smem_ptr<BType*>>, TileLayoutTypeB>;

    TileTensorTypeA &a_tensor;
    TileTensorTypeB &b_Tensor;

    bool is_leader{false};

    private:
    static constexpr size_t transaction_bytes{
        (cosize_v<TileLayoutTypeA> * sizeof(AType) + cosize_v<TileLayoutTypeB> * sizeof(BType)) / pipeline_stages
    };

    public:
    COBRA_DEVICE UMMAConsumerView(
        TileTensorTypeA& a_smem_tensor,
        TileTensorTypeB& b_smem_tensor,
        ProducerBarrierArrayType& producer_barrier,
        ConsumerBarrierArrayType& consumer_barrier
    ):
        a_tensor(a_smem_tensor), b_Tensor(b_smem_tensor),
        producer_barrier(producer_barrier), consumer_barrier(consumer_barrier) {
        is_leader = (cute::elect_one_sync() == 1);
    }

    static constexpr size_t arrival_count{1};
    static constexpr size_t static_pipeline_stages{pipeline_stages};

    /**
     * @brief preemptively checks if the barrier needs to be waited on
     * ready to be filled
     * @param state
     * @return BarrierStatus
     */
    COBRA_DEVICE BarrierStatus check_barrier(State& state){
        return static_cast<BarrierStatus>(producer_barrier[state.index].try_wait(state.phase));
    }

    /**
     * @brief gets the shared memory tiles once they are available 
     * 
     * @param state encapsulates the stage in the pipeline we are on
     * @param status whether to skip the wait or not
     * @return a tuple of the a and b slice 
     */
    COBRA_DEVICE auto get_tiles(State& state, BarrierStatus status){
        if (status == BarrierStatus::WaitAgain) 
            producer_barrier[state.index].wait(state.phase);

        return make_tuple(
            a_tensor(_, _, _, state.index), 
            b_Tensor(_, _, _, state.index)
        );
    }

    COBRA_DEVICE void commit_work(State& state){
        uint64_t* smem_ptr{reinterpret_cast<uint64_t*>(&consumer_barrier[state.index])};
        cutlass::arch::umma_arrive(smem_ptr);
    }

    COBRA_S_DEVICE State initial_state() {
        return State{0, 0};
    }
};
}


/**
 * @brief A 2 way barrier. Consists of a full barrier which 
 * a producer signals when it is done writing data,
 * and a empty barrier signalled by a consumer when its 
 * done reading data. The producer waits on the empty barrier
 * to write data, and the consumer waits on the full barrier.
 * Synchronization pattern and initialization works best when
 * used with the TMA.
 * 
 * @tparam ProducerView a view of the pipeline with respect to the producers 
 * responsibilities (cp async, tma, etc)
 * @tparam ConsumerView a view of the pipeline with respect to the consumers 
 * responsibilities (umma, wgmma, mma)
 * @tparam ThreadRoleType typically a enum with various supported thread roles
 */
template<
    PipelineViewType ProducerView,
    PipelineViewType ConsumerView,
    ThreadRoleCompatible ThreadRoleType
>
struct TwoWayPipeline {

    static_assert(std::is_same_v<typename ProducerView::TileTensorTypeA, typename ConsumerView::TileTensorTypeA>,
        "ProducerView and ConsumerView must agree on TileTensorTypeA");
    static_assert(std::is_same_v<typename ProducerView::TileTensorTypeB, typename ConsumerView::TileTensorTypeB>,
        "ProducerView and ConsumerView must agree on TileTensorTypeB");
    static_assert(std::is_same_v<typename ProducerView::ProducerBarrierArrayType, typename ConsumerView::ProducerBarrierArrayType>,
        "ProducerView and ConsumerView must agree on ProducerBarrierArrayType");
    static_assert(std::is_same_v<typename ProducerView::ConsumerBarrierArrayType, typename ConsumerView::ConsumerBarrierArrayType>,
        "ProducerView and ConsumerView must agree on ConsumerBarrierArrayType");
    static_assert(ProducerView::static_pipeline_stages == ConsumerView::static_pipeline_stages,
        "ProducerView and ConsumerView must agree on pipeline_stages");

    using ProducerBarrierArrayType = typename ProducerView::ProducerBarrierArrayType;
    using ConsumerBarrierArrayType = typename ConsumerView::ConsumerBarrierArrayType;

    using TileTensorTypeA = typename ProducerView::TileTensorTypeA;
    using TileTensorTypeB = typename ProducerView::TileTensorTypeB;

    ProducerBarrierArrayType &producer_barrier;
    ConsumerBarrierArrayType &consumer_barrier;

    const ThreadRoleType producer_role; 
    const ThreadRoleType consumer_role;

    const ThreadRoleType role;

    /**
     * @brief initializes the barrier
     * 
     * @param producer_barrier typically an array of uint64_t in smem
     * @param consumer_barrier typically an array of uint64_t in smem
     * @param producer_role  which role corresponds to the producer (loading data)
     * @param consumer_role  which role corresponds to the consumer (reading data)
     * @param role the role of this thread
     */
    COBRA_DEVICE TwoWayPipeline(
        ProducerBarrierArrayType &producer_barrier,
        ConsumerBarrierArrayType &consumer_barrier,
        const ThreadRoleType producer_role,
        const ThreadRoleType consumer_role,
        const ThreadRoleType& role
    ): 
            producer_barrier(producer_barrier), consumer_barrier(consumer_barrier),
            producer_role(producer_role), consumer_role(consumer_role), 
            role(role){

        if (role == producer_role) {
            if (elect_one_sync()) {
                for (size_t i{0}; i < ProducerView::static_pipeline_stages; ++i) {
                    producer_barrier[i].init(ProducerView::arrival_count); 
                    consumer_barrier[i].init(ConsumerView::arrival_count); 
                }
            }
        }

        // syncthreads does not guarantee the visibility of barrier initialization among all threads,
        // fence_barrier_init() does. So we add a syncthreads after the fence to ensure all threads wait
        // for thread0 to init and fence the instruction before proceeding.
        cutlass::arch::fence_barrier_init();
        __syncthreads();
    }

    /**
     * @brief Get the producer view object
     * 
     * @param a_tensor pipelined smem tensor A
     * @param b_tensor pipelined smem tensor B
     * @return ProducerView 
     */
    COBRA_DEVICE ProducerView get_producer_view(TileTensorTypeA &a_tensor, TileTensorTypeB &b_tensor) {
        return ProducerView(a_tensor, b_tensor, producer_barrier, consumer_barrier);
    }

    /**
     * @brief Get the consumer view object
     * 
     * @param a_tensor pipelined smem tensor A
     * @param b_tensor pipelined smem tensor B
     * @return ConsumerView 
     */
    COBRA_DEVICE ConsumerView get_consumer_view(TileTensorTypeA &a_tensor, TileTensorTypeB &b_tensor) {
        return ConsumerView(a_tensor, b_tensor, producer_barrier, consumer_barrier);
    }
};

}