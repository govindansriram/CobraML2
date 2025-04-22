//
// Created by sriram on 12/15/24.
//

#include "allocator.h"
#include "standard_kernel/standard_allocator.h"
#include <array>  // Add this line
#include "cuda_allocator.h"

namespace cobraml::core {

    std::array<std::unique_ptr<Allocator>, 4> global_allocators{
        std::make_unique<StandardAllocator>(),
        std::make_unique<CudaAllocator>(),
        std::make_unique<StandardAllocator>(),
        std::make_unique<StandardAllocator>(),
    };

    Allocator * get_allocator(Device const device) {
        return global_allocators[device].get();
    }

}
