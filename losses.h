#pragma once

#include "function.h"

namespace boostnet {

class mse : public function {
public:
    mse(function_ptr x, function_ptr y)
        : function(2, 1)
    {
        if (x->get_output_count() != y->get_output_count()) {
            throw std::logic_error("arguments of different sizes");
        }

        prerequisites_.at(0) = x;
        prerequisites_.at(1) = y;
    }

public:
    virtual function_ptr clone() const override {
        return std::make_shared<mse>(*this);
    }

    virtual void forward(computation_context& context) override;
    virtual void backward_x(computation_context& context) override;

private:
    static matrix& x_gradient_cache(computation_context& context) {
        return context.local_data.at(0);
    }

    static matrix& y_gradient_cache(computation_context& context) {
        return context.local_data.at(1);
    }

    static matrix& diff_cache(computation_context& context) {
        return context.local_data.at(2);
    }
};

inline function_ptr make_mse(function_ptr x, function_ptr y) {
    return std::make_shared<mse>(x, y);
}

}
