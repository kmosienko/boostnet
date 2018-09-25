#include "model.h"

#include <limits>
#include <algorithm>

namespace boostnet {

model::model(function_ptr node) {
    size_t time = 0;
    std::vector<size_t> times;

    if (dfs(node, time, times) != 0) {
        throw std::logic_error("wrong algorithm");
    }
    if (times.size() != nodes_.size()) {
        throw std::logic_error("wrong algorithm");
    }
    if (prerequisites_.size() != nodes_.size()) {
        throw std::logic_error("wrong algorithm");
    }
    if (usages_.size() != nodes_.size()) {
        throw std::logic_error("wrong algorithm");
    }
    for (const size_t t : times) {
        if (t == std::numeric_limits<size_t>::max()) {
            throw std::logic_error("wrong algorithm");
        }
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        computation_order_.push_back(i);
    }
    std::sort(computation_order_.begin(), computation_order_.end(),
              [&times](size_t i, size_t j) {
        size_t t_i = times.at(i);
        size_t t_j = times.at(j);

        if (t_i == t_j) {
            throw std::logic_error("wrong algorithm");
        }

        return t_i < t_j;
    });

    contexts_.resize(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const function_ptr& node = nodes_.at(i);
        const std::vector<size_t>& prerequisites = prerequisites_.at(i);

        if (node->prerequisites().size() != prerequisites.size()) {
            throw std::logic_error("wrong algorithm");
        }
        for (size_t j = 0; j < prerequisites.size(); ++j) {
            if (prerequisites.at(j) != ptr_to_index_.at(node->prerequisites().at(j).get())) {
                throw std::logic_error("wrong algorithm");
            }
        }

        computation_context& current_context = contexts_.at(i);
        current_context.input.resize(prerequisites.size(), nullptr);
        current_context.input_gradients.resize(prerequisites.size(), nullptr);

        for (size_t j = 0; j < prerequisites.size(); ++j) {
            computation_context& prerequisite_context = contexts_.at(prerequisites.at(j));

            current_context.input.at(j) = &prerequisite_context.output;
            current_context.input_gradients.at(j) = &prerequisite_context.output_gradient;
        }
    }

    for (computation_context& context : contexts_) {
        for (const auto* p : context.input) {
            if (p == nullptr) {
                throw std::logic_error("wrong algorithm");
            }
        }
        for (const auto* p : context.input_gradients) {
            if (p == nullptr) {
                throw std::logic_error("wrong algorithm");
            }
        }
    }
}


void model::forward() {
    std::vector<bool> done(nodes_.size(), false);

    for (const size_t i : computation_order_) {
        for (const size_t j : prerequisites_.at(i)) {
            if (!done.at(j)) {
                throw std::logic_error("wrong algorithm");
            }
        }
        for (const size_t j : usages_.at(i)) {
            if (done.at(j)) {
                throw std::logic_error("wrong algorithm");
            }
        }

        const function_ptr& node = nodes_.at(i);
        computation_context& context = contexts_.at(i);

        node->forward(context);
        context.output_gradient.setZero(
                    context.output.rows(),
                    context.output.cols());

        done.at(i) = true;
    }

    for (const bool d : done) {
        if (!d) {
            throw std::logic_error("wrong algorithm");
        }
    }
}

void model::backward(const matrix& output_gradient,
                     const bool optimize_constantness)
{
    if (nodes_.empty()) {
        return;
    }

    std::vector<bool> done(nodes_.size(), false);

    {
        computation_context& context = contexts_.at(computation_order_.back());
        if (is_not_initialized(output_gradient)) {
            if (context.output_gradient.rows() != 1 ||
                context.output_gradient.cols() != 1)
            {
                throw std::logic_error("wrong algorithm");
            }
            context.output_gradient(0, 0) = 1;
        } else {
            context.output_gradient = output_gradient;
        }
    }

    for (const size_t i : reversed(computation_order_)) {
        for (const size_t j : prerequisites_.at(i)) {
            if (done.at(j)) {
                throw std::logic_error("wrong algorithm");
            }
        }
        for (const size_t j : usages_.at(i)) {
            if (!done.at(j)) {
                throw std::logic_error("wrong algorithm");
            }
        }

        const function_ptr& node = nodes_.at(i);
        computation_context& context = contexts_.at(i);

        if (!optimize_constantness || !node->all_prerequisites_are_constant()) {
            node->backward_x(context);
        }
        if (!optimize_constantness || !node->is_constant()) {
            node->backward_w(context);
        }

        done.at(i) = true;
    }

    for (const bool d : done) {
        if (!d) {
            throw std::logic_error("wrong algorithm");
        }
    }
}

void model::tune(const float learning_rate) {
    std::vector<size_t> indexes_for_tune;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const function_ptr& node = nodes_.at(i);

        if (!node->is_constant()) {
            indexes_for_tune.push_back(i);
        }
    }

    const size_t iterations = indexes_for_tune.size();
#pragma omp parallel for
    for (size_t j = 0; j < iterations; ++j) {
        const function_ptr& node = nodes_.at(indexes_for_tune[j]);
        computation_context& context = contexts_.at(indexes_for_tune[j]);

        if (!node->is_constant()) {
            node->tune(context, learning_rate);
        }
    }
}

size_t model::dfs(const function_ptr& node, size_t& time, std::vector<size_t>& times) {
    if (ptr_to_index_.find(node.get()) != ptr_to_index_.end()) {
        return ptr_to_index_.at(node.get());
    }

    const size_t index = nodes_.size();
    nodes_.push_back(node);
    ptr_to_index_[node.get()] = index;

    if (prerequisites_.size() < index + 1) {
        prerequisites_.resize(index + 1);
    }
    if (usages_.size() < index + 1) {
        usages_.resize(index + 1);
    }

    for (const function_ptr& dep : node->prerequisites()) {
        const size_t dep_index = dfs(dep, time, times);

        if (prerequisites_.size() < dep_index + 1) {
            prerequisites_.resize(dep_index + 1);
        }
        if (usages_.size() < dep_index + 1) {
            usages_.resize(dep_index + 1);
        }

        prerequisites_.at(index).push_back(dep_index);
        usages_.at(dep_index).push_back(index);
    }

    if (times.size() < index + 1) {
        times.resize(index + 1, std::numeric_limits<size_t>::max());
    }
    times.at(index) = time;
    ++time;

    return index;
}

}
