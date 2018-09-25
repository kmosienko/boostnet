#pragma once

#include "function.h"

#include <exception>
#include <stdexcept>
#include <vector>
#include <unordered_map>

namespace boostnet {

class model {
public:
    model(function_ptr node);

public:
    template<typename T>
    model& set_input(const function_ptr& node, const T& value) {
        if (!node->is_constant()) {
            throw std::logic_error("wrong algorithm");
        }

        const size_t index = ptr_to_index_.at(node.get());
        computation_context& context = contexts_.at(index);

        if (!context.input.empty() || !context.input_gradients.empty()) {
            throw std::logic_error("wrong algorithm");
        }

        context.output = value;

        return *this;
    }

    template<typename T>
    model& set_input(const std::vector<std::pair<const function_ptr&, const T&>>& model_input) {
        for (const std::pair<const function_ptr&, const matrix&>& p : model_input) {
            set_input(p.first, p.second);
        }

        return *this;
    }

    template<typename T>
    model& move_input(const function_ptr& node, T&& value) {
        if (!node->is_constant()) {
            throw std::logic_error("wrong algorithm");
        }

        const size_t index = ptr_to_index_.at(node.get());
        computation_context& context = contexts_.at(index);

        if (!context.input.empty() || !context.input_gradients.empty()) {
            throw std::logic_error("wrong algorithm");
        }

        context.output = std::move(value);

        return *this;
    }

    const matrix& get_output(const function_ptr& node) const {
        const size_t index = ptr_to_index_.at(node.get());
        const computation_context& context = contexts_.at(index);

        return context.output;
    }

    const matrix& get_output() const {
        const computation_context& context = contexts_.at(0);

        return context.output;
    }

    scalar_type get_plain_output() const {
        const matrix& output = get_output();

        if (output.rows() != 1 || output.cols() != 1) {
            throw std::logic_error("wrong model output");
        }

        return output(0, 0);
    }

    const matrix& get_output_gradient(const function_ptr& node) const {
        const size_t index = ptr_to_index_.at(node.get());
        const computation_context& context = contexts_.at(index);

        return context.output_gradient;
    }

    void forward();
    void backward(const matrix& output_gradient = matrix(),
                  const bool optimize_constantness = true);
    void tune(const float learning_rate);

private:
    size_t dfs(const function_ptr& node, size_t& time, std::vector<size_t>& times);

private:
    std::vector<function_ptr>             nodes_;
    std::unordered_map<function*, size_t> ptr_to_index_;
    std::vector<size_t>                   computation_order_;
    std::vector<std::vector<size_t>>      prerequisites_;
    std::vector<std::vector<size_t>>      usages_;
    std::vector<computation_context>      contexts_;
};

template<typename... T>
matrix evaluate(function_ptr original_function, const T&... x) {
    std::vector<const matrix*> input = {&x...};
    if (input.size() != original_function->prerequisites().size()) {
        throw std::logic_error("bad parameter count");
    }

    function_ptr cloned_function = original_function->clone();
    function* f = cloned_function.get();

    for (size_t i = 0; i < f->prerequisites().size(); ++i) {
        f->prerequisite(i) = make_constant(input.at(i)->cols());
    }

    model m(cloned_function);
    for (size_t i = 0; i < f->prerequisites().size(); ++i) {
        m.set_input(f->prerequisite(i), *input.at(i));
    }
    m.forward();

    return m.get_output();
}

}
