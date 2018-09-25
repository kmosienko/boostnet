#include "losses.h"

namespace boostnet {

void mse::forward(computation_context& context) {
    context.local_data.resize(3);

    matrix& diff = diff_cache(context);
    diff.resizeLike(x_value(context));
    diff.noalias() = x_value(context) - y_value(context);

    if (!x()->is_constant()) {
        x_gradient_cache(context).noalias() = (2.0f * diff/* / ((scalar_type)diff.rows())*/);

        if (!y()->is_constant()) {
            y_gradient_cache(context).noalias() = -x_gradient_cache(context);
        }
    } else if (!y()->is_constant()) {
        y_gradient_cache(context).noalias() = -(2.0f * diff/* / ((scalar_type)diff.rows())*/);
    }

    diff.array() *= diff.array();
    context.output.noalias() = diff.rowwise().sum().colwise().mean();
}

void mse::backward_x(computation_context& context) {
    if (context.output_gradient.rows() != 1 || context.output_gradient.cols() != 1) {
        throw std::logic_error("bad output_gradient");
    }

    const auto out_gradient = context.output_gradient(0, 0);

    if (!x()->is_constant()) {
        x_gradient(context) += (x_gradient_cache(context) * out_gradient);

        if (!y()->is_constant()) {
            y_gradient(context) -= x_gradient(context);
        }
    } else if (!y()->is_constant()) {
        y_gradient(context) += (y_gradient_cache(context) * out_gradient);
    }
}

}
