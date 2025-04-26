//
// Created by Sriram Govindan on 12/24/24.
//

#include "math_dis.h"
#include "standard_math.h"
#include <array>

namespace cobraml::core {
    std::array<std::unique_ptr<Math>, 3> global_math_kernels = {
        std::make_unique<StandardMath>(),
        std::make_unique<StandardMath>(),
        std::make_unique<StandardMath>(),
    };

    Math * get_math_kernels(const Device device) {
        return global_math_kernels[device].get();
    }
}
