#pragma once

#include "function.h"

namespace boostnet {

class linear : public function {
public:
    linear(function_ptr x, size_t output_count);

public:
    virtual function_ptr clone() const override {
        return std::make_shared<linear>(*this);
    }

    virtual void forward(computation_context& context) override;
    virtual void backward_x(computation_context& context) override;
    virtual void backward_w(computation_context& context) const override;
    virtual void tune(computation_context& context, const float learning_rate) override;

private:
    static matrix& a_gradient(computation_context& context) {
        return context.local_data.at(0);
    }

    static matrix& b_gradient(computation_context& context) {
        return context.local_data.at(1);
    }

private:
    matrix A_;
    row_vector b_;
};

inline function_ptr make_linear(function_ptr x, size_t output_count) {
    return std::make_shared<linear>(x, output_count);
}



class sigmoid : public function {
public:
    sigmoid(function_ptr x)
        : function(1, x->get_output_count())
    {
        prerequisites_.at(0) = x;
    }

public:
    virtual function_ptr clone() const override {
        return std::make_shared<sigmoid>(*this);
    }

    virtual void forward(computation_context& context) override;
    virtual void backward_x(computation_context& context) override;
};

inline function_ptr make_sigmoid(function_ptr x) {
    return std::make_shared<sigmoid>(x);
}



class dropout : public function {
public:
    dropout(function_ptr x, float p)
        : function(1, x->get_output_count())
        , p_(p)
    {
        prerequisites_.at(0) = x;
    }

public:
    virtual function_ptr clone() const override {
        return std::make_shared<dropout>(*this);
    }

    virtual void forward(computation_context& context) override;
    virtual void backward_x(computation_context& context) override;

private:
    const float p_;
    array       mask_;
};

inline function_ptr make_dropout(function_ptr x, float p) {
    return std::make_shared<dropout>(x, p);
}

}
