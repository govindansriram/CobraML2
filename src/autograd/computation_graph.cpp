//
// Created by sriram on 4/3/25.
//

#include "computation_graph.h"

#include <functional>
#include <iostream>
#include <set>
#include <stack>

#include "brarray.h"

namespace cobraml::core {

    AutogradNode::AutogradNode(const std::initializer_list<std::shared_ptr<AutogradNode> > &previous_nodes, const bool retain_grad): prev_nodes(previous_nodes), retain_grad(retain_grad){}

    void AutogradNode::add_next_node(const std::shared_ptr<AutogradNode> &node, int key) {
        next_nodes.emplace_back(node, key);
    }

    void AutogradNode::accumulate_gradients(Brarray &local_gradient) {
        if (next_nodes.empty()) {
            return;
        }

        for (const auto &[w_ptr, key]: next_nodes) {
            if (const auto shared = w_ptr.lock()) {
                iadd(local_gradient, shared->get_gradient(key));
            }else {
                std::cerr << "Warning required computational graph node is not available, graph is corrupted";
            }
        }
    }


    std::vector<std::shared_ptr<AutogradNode>> topological_sort(const std::shared_ptr<AutogradNode> &head) {
        std::stack<std::shared_ptr<AutogradNode>> ptr_stack;
        std::set<uintptr_t> ptr_set; // track visited nodes
        std::vector<std::shared_ptr<AutogradNode>> ret;

        std::function<void (const std::shared_ptr<AutogradNode>&)> dfs = [&](const std::shared_ptr<AutogradNode> &node) {
            std::vector<std::weak_ptr<AutogradNode>> explore;

            const uintptr_t addr{reinterpret_cast<uintptr_t>(node.get())}; // see if ptr has already been visited
            ptr_set.insert(addr);

            for (const std::shared_ptr<AutogradNode> &prev: node->prev_nodes) { // track all the children
                dfs(prev);
            }

            ptr_stack.push(node);
        };

        dfs(head);

        while (!ptr_stack.empty()) {
            ret.push_back(ptr_stack.top());
            ptr_stack.pop();
        }

        return ret;
    }

    void back_propagate(const std::shared_ptr<ActivationNode> &activation_node) {

        // back_propagation
        std::vector order{topological_sort(activation_node)};
        for (const std::shared_ptr<AutogradNode> &ptr : order) {
            ptr->compute_backwards_gradients();
        }

        // release the graph
        for (size_t i{order.size()}; i > 0; --i) {
            order[i - 1]->release_gradients();
            order[i - 1]->next_nodes = {};
            order[i - 1]->prev_nodes = {};
        }
    }

    ActivationNode::ActivationNode(
        const Brarray &activation,
        const std::initializer_list<std::shared_ptr<AutogradNode>> & ptrs,
        const bool retain_gradients):
    AutogradNode(ptrs, retain_gradients), activation(activation.shared_copy()){}

    const Brarray &ActivationNode::get_gradient(int) const{
        return gradient;
    }

    void ActivationNode::compute_backwards_gradients() {
        gradient = Brarray(activation.get_device(), activation.get_dtype(), activation.get_shape());
        iadd(gradient, 1);
        accumulate_gradients(gradient);
    }

    Brarray &ActivationNode::get_gradient() {
        return gradient;
    }

    void ActivationNode::release_gradients() {
        if (!retain_grad) {
            gradient = Brarray();
        }
    }



}

