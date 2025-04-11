//
// Created by sriram on 4/11/25.
//

#include "element_wise.h"

#include <iostream>

namespace cobraml::core {
    std::vector<std::shared_ptr<AutogradNode> > register_inputs(
        const std::shared_ptr<ActivationNode> &node_x,
        const std::shared_ptr<ActivationNode> &node_y) {
        std::vector<std::shared_ptr<AutogradNode> > ret;
        ret.reserve(2);
        if (node_x != nullptr) {
            ret.push_back(node_x);
        }

        if (node_y != nullptr) {
            ret.push_back(node_y);
        }

        if (ret.empty()) throw std::runtime_error("no preceding nodes provided");

        return ret;
    }

    ElementWise::ElementWise(
        const Brarray &activation_x,
        const Brarray &activation_y,
        const std::shared_ptr<ActivationNode> &node_x,
        const std::shared_ptr<ActivationNode> &node_y): AutogradNode(register_inputs(node_x, node_y),
                                                                     false),
                                                        activation_x(activation_x.shared_copy()),
                                                        activation_y(activation_y.shared_copy()),
                                                        x_requires_grads(node_x != nullptr),
                                                        y_requires_grads(node_y != nullptr) {}

    const Brarray &ElementWise::get_gradient(const int key) const {
        if (key == 0 && x_requires_grads) {
            return gradient_x;
        }

        if (key == 1 && y_requires_grads) {
            return gradient_y;
        }

        throw std::runtime_error("invalid key provided");
    }

    void ElementWise::release_gradients() {
        if (!retain_grad) {
            gradient_x = Brarray();
            gradient_y = Brarray();
        }
    }

    void Multiply::compute_backwards_gradients() {
        // TODO handle shapes beyond scalars

        Brarray acc(activation_x.get_device(), activation_x.get_dtype(), activation_x.get_shape());
        iadd(acc, 1);
        accumulate_gradients(acc);

        if (x_requires_grads) {
            gradient_x = mult(acc, activation_y, false);

            // TODO broadcast down
        }

        if (y_requires_grads) {
            gradient_y = mult(acc, activation_x, false);

            // TODO broadcast down
        }
    }

}
