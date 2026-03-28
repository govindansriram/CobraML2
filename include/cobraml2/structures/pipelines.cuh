#pragma once
#include <concepts>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include "../macros.cuh"
#include "../utilities.cuh"
#include "./load.cuh"


namespace cobraml::pipelines{
using namespace cute;

template<typename V>
concept PipelineViewType = requires {
    typename V::ProducerBarrierArrayType;
    typename V::ConsumerBarrierArrayType;
    { V::static_arrival_count } -> std::convertible_to<size_t>;
    { V::static_pipeline_stages } -> std::convertible_to<size_t>;
} && std::constructible_from<
    V,
    typename V::ProducerBarrierArrayType&,
    typename V::ConsumerBarrierArrayType&
>;

template<typename T>
concept ThreadRoleCompatible = std::equality_comparable<T>;

struct State {
    int index{0};
    uint32_t phase{0};
    size_t pipeline_stages{0};
    size_t counter{0};

    COBRA_DEVICE State& operator++() {
        ++index;
        ++counter;
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

namespace sm100{
template<
    typename AType,
    typename BType,
    size_t pipeline_stages,
    size_t arrival_count
>
struct ProducerView {

    using LoadTypeA = loadop::sm100::TMALoad<AType, true>;
    using LoadTypeB = loadop::sm100::TMALoad<BType, false>;

    using ProducerBarrierType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarrierType = cutlass::arch::ClusterBarrier;

    using ProducerBarrierArrayType = ProducerBarrierType[pipeline_stages];
    using ConsumerBarrierArrayType = ConsumerBarrierType[pipeline_stages];

    ProducerBarrierArrayType &producer_barrier;
    ConsumerBarrierArrayType &consumer_barrier;

    bool is_leader{false};
    static constexpr size_t static_arrival_count{arrival_count};
    static constexpr size_t static_pipeline_stages{pipeline_stages};

    private:

    template<typename SmemTensorAType, typename SmemTensorBType>
    struct TileIterator{
        SmemTensorAType &a_tensor;
        SmemTensorBType &b_tensor;

        COBRA_DEVICE TileIterator(SmemTensorAType &a_tensor, SmemTensorBType &b_tensor): a_tensor(a_tensor), b_tensor(b_tensor){}

        COBRA_DEVICE auto next(State& state, BarrierStatus status){
            if (status == BarrierStatus::WaitAgain)
                consumer_barrier[state.index].wait(state.phase);

            return make_tuple(
                slice_last(a_tensor, state.index),
                slice_last(b_tensor, state.index),
                &producer_barrier[state.index]
            );
        }
    };

    public:

    COBRA_DEVICE ProducerView(
        ProducerBarrierArrayType& producer_barrier,
        ConsumerBarrierArrayType& consumer_barrier
    ): producer_barrier(producer_barrier), consumer_barrier(consumer_barrier) {
        is_leader = (cute::elect_one_sync() == 1);
    }

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
     * @brief 
     * 
     * @tparam SmemTensorAType 
     * @tparam SmemTensorBType 
     * @param a_tensor 
     * @param b_tensor 
     * @return a iterator over all available tiles 
     */
    template<typename SmemTensorAType, typename SmemTensorBType>
    COBRA_DEVICE auto make_tile_iterator(SmemTensorAType &a_tensor, SmemTensorBType &b_tensor) {
        return TileIterator<SmemTensorAType, SmemTensorBType>(a_tensor, b_tensor);
    }

    COBRA_S_DEVICE State initial_state() {
        return State{0, 1};
    }
};


template<
    typename AType,
    typename BType,
    size_t pipeline_stages,
    size_t arrival_count
>
struct ConsumerView {

    using ProducerBarrierType = cutlass::arch::ClusterTransactionBarrier;
    using ConsumerBarrierType = cutlass::arch::ClusterBarrier;

    using ProducerBarrierArrayType = ProducerBarrierType[pipeline_stages];
    using ConsumerBarrierArrayType = ConsumerBarrierType[pipeline_stages];

    ProducerBarrierArrayType &producer_barrier;
    ConsumerBarrierArrayType &consumer_barrier;

    bool is_leader{false};

    static constexpr size_t static_arrival_count{arrival_count};
    static constexpr size_t static_pipeline_stages{pipeline_stages};

    private:

    template<typename SmemTensorAType, typename SmemTensorBType>
    struct TileIterator {
        SmemTensorAType &a_tensor;
        SmemTensorBType &b_tensor;

        COBRA_DEVICE TileIterator(SmemTensorAType &a_tensor, SmemTensorBType &b_tensor): a_tensor(a_tensor), b_tensor(b_tensor){}

        COBRA_DEVICE auto next(State& state, BarrierStatus status){
            if (status == BarrierStatus::WaitAgain)
                producer_barrier[state.index].wait(state.phase);

            return make_tuple(
                slice_last(a_tensor, state.index),
                slice_last(b_tensor, state.index),
                &consumer_barrier[state.index]
            );
        }
    };

    public:

    COBRA_DEVICE ConsumerView(
        ProducerBarrierArrayType& producer_barrier,
        ConsumerBarrierArrayType& consumer_barrier
    ): producer_barrier(producer_barrier), consumer_barrier(consumer_barrier) {
        is_leader = (cute::elect_one_sync() == 1);
    }

    /**
     * @brief preemptively checks if the barrier needs to be waited on
     * ready to be filled
     * @param state
     * @return BarrierStatus
     */
    COBRA_DEVICE BarrierStatus check_barrier(State& state){
        return static_cast<BarrierStatus>(producer_barrier[state.index].try_wait(state.phase));
    }

    template<typename SmemTensorAType, typename SmemTensorBType>
    COBRA_DEVICE auto make_tile_iterator(SmemTensorAType &a_tensor, SmemTensorBType &b_tensor) {
        return TileIterator<SmemTensorAType, SmemTensorBType>(a_tensor, b_tensor);
    }

    COBRA_S_DEVICE State initial_state() {
        return State{0, 0};
    }
};
}


/**
 * @brief A 2 way pipeline. The producer barrier is signalled when data
 * has been written (e.g. TMA completes), the consumer barrier is signalled
 * when data has been consumed. Each view waits on the opposite barrier
 * and returns a pointer to its own barrier slot for the caller to signal.
 *
 * @tparam ProducerView view that loads data, waits on consumer barrier
 * @tparam ConsumerView view that consumes data, waits on producer barrier
 * @tparam ThreadRoleType enum used to assign threads to producer/consumer roles
 */
template<
    PipelineViewType ProducerView,
    PipelineViewType ConsumerView,
    ThreadRoleCompatible ThreadRoleType
>
struct TwoWayPipeline {

    static_assert(std::is_same_v<typename ProducerView::ProducerBarrierArrayType, typename ConsumerView::ProducerBarrierArrayType>,
        "ProducerView and ConsumerView must agree on ProducerBarrierArrayType");
    static_assert(std::is_same_v<typename ProducerView::ConsumerBarrierArrayType, typename ConsumerView::ConsumerBarrierArrayType>,
        "ProducerView and ConsumerView must agree on ConsumerBarrierArrayType");
        
    static_assert(ProducerView::static_pipeline_stages == ConsumerView::static_pipeline_stages,
        "ProducerView and ConsumerView must agree on pipeline_stages");

    using ProducerBarrierArrayType = typename ProducerView::ProducerBarrierArrayType;
    using ConsumerBarrierArrayType = typename ConsumerView::ConsumerBarrierArrayType;

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
                    producer_barrier[i].init(ProducerView::static_arrival_count);
                    consumer_barrier[i].init(ConsumerView::static_arrival_count); 
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
     * @brief Get the producer view, waits on consumer barrier, returns producer barrier slots
     * @return ProducerView
     */
    COBRA_DEVICE ProducerView get_producer_view() {
        return ProducerView(producer_barrier, consumer_barrier);
    }

    /**
     * @brief Get the consumer view, waits on producer barrier, returns consumer barrier slots
     * @return ConsumerView
     */
    COBRA_DEVICE ConsumerView get_consumer_view() {
        return ConsumerView(producer_barrier, consumer_barrier);
    }
};

template<typename AType, typename BType, size_t pipeline_stages>
using UmmaConsumerViewType = ConsumerView<AType, BType, pipeline_stages, 1>;

template<typename AType, typename BType, size_t pipeline_stages>
using TmaProducerViewType = ProducerView<AType, BType, pipeline_stages, 1>;

template<typename ThreadRoleType, typename AType, typename BType, size_t pipeline_stages>
using PipelineTmaUmmaType = TwoWayPipeline<
    TmaProducerViewType<AType, BType, pipeline_stages>, 
    UmmaConsumerViewType<AType, BType, pipeline_stages>, 
    ThreadRoleType
>;

}