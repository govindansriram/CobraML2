//
// Created by sriram on 12/19/24.
//

#include "enums.h"

namespace cobraml::core {
    bool is_float(Dtype const type) {
        return type == FLOAT32 || type == FLOAT64;
    }

    std::string dtype_to_string(Dtype const dtype) {
        switch (dtype) {
            case INT8: return "INT8";
            case INT16: return "INT16";
            case INT32: return "INT32";
            case INT64: return "INT64";
            case FLOAT32: return "FLOAT32";
            case FLOAT64: return "FLOAT64";
            case INVALID: return "INVALID";
        }

        return "";
    }

    std::string device_to_string(const Device device) {
        switch (device) {
            case CPU: return "CPU";
            case CUDA: return "GPU";
            case CPU_X: return "CPU Accelerated";
        }

        return "";
    }


    bool operator<(Dtype const lhs, Dtype const rhs) {

        is_invalid(lhs);
        is_invalid(rhs);

        bool const l_if = is_float(lhs);
        bool const r_if = is_float(rhs);

        if (l_if == r_if) {
            return dtype_to_bytes(lhs) < dtype_to_bytes(rhs);
        }

        if (l_if) {
            return false;
        }

        if (r_if) {
            return true;
        }

        return false;
    }

    unsigned char func_pos = 0;

}
