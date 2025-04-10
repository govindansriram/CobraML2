//
// Created by sriram on 4/3/25.
//

#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H
#include <memory>
#include <vector>

#include "brarray.h"

namespace cobraml::core {
    class Brarray;
    struct ActivationNode;

    struct AutogradNode {
        virtual ~AutogradNode() = default;
        std::vector<std::pair<std::weak_ptr<AutogradNode>, int>> next_nodes;
        std::vector<std::shared_ptr<AutogradNode>> prev_nodes;

    protected:
        virtual const Brarray &get_next_gradient(int key) = 0;
        virtual void compute_gradients() = 0;
        friend void back_propagate(const std::shared_ptr<ActivationNode> &activation_node);
    };

    struct ActivationNode: AutogradNode {
        ActivationNode() = default;
        // void add_next_node(const std::shared_ptr<AutogradNode> &node);
        // void add_prev_node(const std::shared_ptr<AutogradNode> &node);
        const Brarray& get_gradient();

    protected:
        std::unique_ptr<Brarray> gradient{std::make_unique<Brarray>()};
        void compute_gradients() override;
        const Brarray &get_next_gradient(int key) override;
    };

    void back_propagate(const std::shared_ptr<ActivationNode> &activation_node);
}

// TODO single element gemv may fail due to unalligned data
// Test GEMV on indexed tensors

#endif //COMPUTATION_GRAPH_H
