#pragma once
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace cobraml::structuring {

using namespace cute;

 template<
    typename TiledCopyType,
    typename SourceEngineType,
    typename DestinationEngineType,
    typename SourceLayout,
    typename DestinationLayout
  >
  struct SyncTiledWriter{

    using SourceTensorType = Tensor<SourceEngineType, SourceLayout>;
    using DestinationTensorType = Tensor<DestinationEngineType, DestinationLayout>;
    using DType = typename SourceTensorType::value_type;

    const SourceTensorType &source_slice;
    DestinationTensorType &dest_slice;
    const TiledCopyType &tiled_copy; 

    SyncTiledWriter(
      const SourceTensorType &source_tensor, 
      DestinationTensorType &dest_tensor,
      const TiledCopyType &tiled_copy
    ) : source_slice(source_tensor), dest_slice(dest_tensor), tiled_copy(tiled_copy){}

    template <int N>
    __forceinline__ __device__ static auto make_last_coord(int coord) {
        auto underscores{repeat<N-1>(_)};
        return cute::tuple_cat(underscores, cute::make_tuple(coord));
    }

    template <int N>
    __forceinline__ __device__ static auto make_first_coord(int coord) {
        auto underscores{repeat<N-1>(_)};
        return cute::tuple_cat(cute::make_tuple(coord), underscores);
    }

    void copy_tensor(){

      constexpr size_t source_rank{
        rank_v<SourceLayout>;
      };

      constexpr size_t dest_rank{
        rank_v<DestinationLayout>;
      };

      static_assert(source_rank == dest_rank);
      
      copy(tiled_copy, source_slice, dest_slice);
    }

    void copy_tensor(size_t iter){
      constexpr size_t source_rank{
        rank_v(SourceLayout);
      };

      constexpr size_t dest_rank{
        rank_v(DestinationLayout);
      };

      auto coord{
        make_last_coord<source_rank>(iter);
      };

      static_assert(source_rank >= dest_rank);

      if constexpr (source_rank == dest_rank){
        copy(tiled_copy, source_slice(coord), dest_slice(coord));
      }else{
        copy(tiled_copy, source_slice(coord), dest_slice);
      }
    }

    template<
      typename IdentityTensorEngineType, 
      typename IdentityTensorLayoutType,
    >
    void predicate_copy_tensor(
      DType fill_value, 
      Tensor<IdentityTensorEngineType, IdentityTensorLayoutType> identity_tensor, 
      int bound){
        predicate_copy_tensor_(
          identity_tensor,
          source_slice,
          dest_slice,
          fill_value,
          bound
        )
      };


    private:
      template<
        typename IdentityTensorEngineType, 
        typename IdentityTensorLayoutType,
        typename SourceTensorEngineType, 
        typename SourceTensorEngineLayoutType,
        typename DestinationTensorEngineType, 
        typename DestinationTensorLayoutType,
      >
      void predicate_copy_tensor_(
        const Tensor<IdentityTensorEngineType, IdentityTensorLayoutType> &identity_tensor, 
        const Tensor<SourceTensorEngineType, SourceTensorEngineLayoutType> &source_tensor, 
        Tensor<DestinationTensorEngineType, DestinationTensorLayoutType> &destination_tensor,
        DType fill_value, 
        int bound){

        constexpr size_t source_rank{
          rank_v(SourceLayout)
        };

        constexpr size_t dest_rank{
          rank_v(DestinationLayout)
        };

        static_assert(source_rank == dest_rank);

        int rows{size(SourceTensorEngineLayoutType{})};

        for (int i{0}; i < rows; ++i){
          auto idx{get<0>(identity_tensor(i))};
          auto coord{make_first_coord<source_rank>(i)};

          if (idx < bound){
            copy(
              tiled_copy, 
              source_tensor(coord), 
              destination_tensor(coord)
            );
          }else{
            fill(destination_tensor(coord), fill_value);
          }
        }
      }
  };

}