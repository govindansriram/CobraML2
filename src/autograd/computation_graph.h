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
        std::vector<std::pair<std::weak_ptr<AutogradNode>, int>> next_nodes{};
        std::vector<std::shared_ptr<AutogradNode>> prev_nodes;
        bool retain_grad;

        AutogradNode(const std::initializer_list<std::shared_ptr<AutogradNode>>& previous_nodes, bool retain_grad);
        virtual void add_next_node(const std::shared_ptr<AutogradNode> &node, int key);
        virtual ~AutogradNode() = default;

    protected:
        virtual void accumulate_gradients(Brarray &local_gradient);
        [[nodiscard]] virtual const Brarray &get_gradient(int key) const = 0;
        virtual void compute_backwards_gradients() = 0;
        virtual void release_gradients() = 0;
        friend void back_propagate(const std::shared_ptr<ActivationNode> &activation_node);
    };

    struct ActivationNode final : AutogradNode {
        Brarray gradient{};
        Brarray activation;

        ActivationNode(
            const Brarray &activation,
            const std::initializer_list<std::shared_ptr<AutogradNode>> &ptrs,
            bool retain_gradients);
        Brarray& get_gradient();

    protected:
        void compute_backwards_gradients() override;
        [[nodiscard]] const Brarray &get_gradient(int) const override;
        void release_gradients() override;
    };

    void back_propagate(const std::shared_ptr<ActivationNode> &activation_node);
}


#endif //COMPUTATION_GRAPH_H
