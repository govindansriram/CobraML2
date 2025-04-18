//
// Created by sriram on 4/11/25.
//

#ifndef ELEM_WISE_H
#define ELEM_WISE_H
#include "computation_graph.h"

namespace cobraml::core {
    struct ElementWise : AutogradNode {
        ElementWise(const Brarray &activation_x,
                    const Brarray &activation_y,
                    const std::shared_ptr<ActivationNode> &node_x,
                    const std::shared_ptr<ActivationNode> &node_y);

    protected:
        Brarray activation_x;
        Brarray activation_y;
        Brarray gradient_x{};
        Brarray gradient_y{};
        bool x_requires_grads;
        bool y_requires_grads;

        [[nodiscard]] const Brarray &get_gradient(int key) const override;

        void release_gradients() override;
    };

    struct Multiply final : ElementWise {
        using ElementWise::ElementWise;
        void compute_backwards_gradients() override;
    };

    struct Add final : ElementWise {
        using ElementWise::ElementWise;
        void compute_backwards_gradients() override;
    };
}


#endif //ELEM_WISE_H
