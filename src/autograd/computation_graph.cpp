//
// Created by sriram on 4/3/25.
//

#include "computation_graph.h"

#include <functional>
#include <set>
#include <stack>

#include "brarray.h"

namespace cobraml::core {

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
            ptr->compute_gradients();
        }

        // release the graph
        for (size_t i{order.size()}; i > 0; --i) {
            order[i - 1]->next_nodes = {};
            order[i - 1]->prev_nodes = {};
        }
    }


}

