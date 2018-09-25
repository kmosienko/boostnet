#include "dense.h"
#include "random.h"

#include <iostream>

namespace boostnet {

linear::linear(function_ptr x, size_t output_count)
    : function(1, output_count)
{
    prerequisites_.at(0) = x;

    A_ = fill_normal<matrix>(x->get_output_count(), output_count_, 0, 0.01);
    b_ = fill_normal<row_vector>(1, output_count_, 0, 0.01);
}

void linear::forward(computation_context& context) {
    context.local_data.resize(2);

    context.output.noalias() = (x_value(context) * A_).rowwise() + b_;
}

void linear::backward_x(computation_context& context) {
    x_gradient(context).noalias() += (context.output_gradient * A_.transpose());
}

void linear::backward_w(computation_context& context) const {
    a_gradient(context).noalias() = x_value(context).matrix().transpose() * context.output_gradient.matrix();
    b_gradient(context).noalias() = context.output_gradient.colwise().sum();
}

void linear::tune(computation_context& context, const float learning_rate) {
    A_.noalias() -= learning_rate * a_gradient(context);
    b_.noalias() -= learning_rate * b_gradient(context);
}



void sigmoid::forward(computation_context& context) {
    context.output = -x_value(context);
    context.output = (1.0f / (1.0f + Eigen::exp(context.output.array()))).matrix();
}

void sigmoid::backward_x(computation_context& context) {
    x_gradient(context) -= context.output_gradient.cwiseProduct(
                (context.output.array() * (context.output.array() - 1.0f)).matrix());
}




void dropout::forward(computation_context& context) {
    if (p_ <= 0) {
        context.output = x_value(context);
        return;
    }

    {
        mask_.setZero(x_value(context).rows(), x_value(context).cols());
        for (auto i = 0; i < mask_.rows(); ++i) {
            for (auto j = 0; j < mask_.cols(); ++j) {
                if (uniform_scalar() >= p_) {
                    mask_(i, j) = 1.0 / (1.0 - p_);
                }
            }
        }
    }

    context.output = x_value(context).array() * mask_;
}

void dropout::backward_x(computation_context& context) {
    if (p_ <= 0) {
        x_gradient(context).noalias() += context.output_gradient;
        return;
    }

    x_gradient(context) += (context.output_gradient.array() * mask_).matrix();
}

}
