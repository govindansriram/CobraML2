#pragma once
#include <concepts>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include "../macros.cuh"

namespace cobraml::structures{
using namespace cute;

template<typename ConfigType>
concept PipelineSmemCompatible = requires {
    typename ConfigType::AType;
    typename ConfigType::BType;
    typename ConfigType::PipelineType;
    { ConfigType::a_smem_layout };
    { ConfigType::b_smem_layout };
    { ConfigType::pipeline_stages } -> std::convertible_to<size_t>;
};

template<PipelineSmemCompatible ConfigType>
struct PipelinedSharedStorage {

    using AType = typename ConfigType::AType;
    using BType = typename ConfigType::BType;
    using PipelineType = typename ConfigType::PipelineType;
    using ProducerBarrierType = typename PipelineType::ProducerBarrierType;
    using ConsumerBarrierType = typename PipelineType::ConsumerBarrierType;

    // Define shared memory buffers for A and B tiles
    alignas(128) cute::ArrayEngine<AType, size(ConfigType::a_smem_layout)> smem_a;
    alignas(128) cute::ArrayEngine<BType, size(ConfigType::b_smem_layout)> smem_b;

    alignas(16) ProducerBarrierType full_barrier[ConfigType::static_pipeline_stages];
    alignas(16) ConsumerBarrierType empty_barrier[ConfigType::static_pipeline_stages];
};


}