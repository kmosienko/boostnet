#include "function.h"

namespace boostnet {

static size_t get_total_output_count(const std::vector<function_ptr>& inputs) {
    size_t res = 0;

    for (const function_ptr& f : inputs) {
        res += f->get_output_count();
    }

    return res;
}

column_stack::column_stack(const std::vector<function_ptr>& inputs)
    : function(inputs.size(),
               get_total_output_count(inputs))
{
    prerequisites_ = inputs;
}

void column_stack::forward(computation_context& context) {
    if (context.input.empty()) {
        return;
    }

    context.output.setZero(context.input.at(0)->rows(), get_output_count());
    for (size_t i = 0; i < prerequisites_.size(); ++i) {
        context.output.col(i) = *context.input.at(i);
    }
}

void column_stack::backward_x(computation_context& context) {
    if (context.input_gradients.empty()) {
        return;
    }

    for (size_t i = 0; i < prerequisites_.size(); ++i) {
        *context.input_gradients.at(i) += context.output_gradient.col(i);
    }
}

}
