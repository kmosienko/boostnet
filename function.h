#pragma once

#include "util.h"

#include <memory>
#include <vector>

namespace boostnet {

class function;
using function_ptr = std::shared_ptr<function>;

struct computation_context {
    std::vector<const matrix*> input;
    matrix                     output;

    std::vector<matrix*>       input_gradients;
    matrix                     output_gradient;

    std::vector<matrix>        local_data;
};

class function : std::enable_shared_from_this<function> {
public:
    explicit function(
            const size_t argument_count,
            const size_t output_count)
        : prerequisites_(argument_count)
        , output_count_(output_count) {}

    virtual ~function() {}

public:
    const std::vector<function_ptr>& prerequisites() const { return prerequisites_; }
    const function_ptr& prerequisite(const size_t index) const { return prerequisites_.at(index); }
    function_ptr& prerequisite(const size_t index) { return prerequisites_.at(index); }

    size_t get_output_count() const { return output_count_; }

    bool all_prerequisites_are_constant() const {
        for (const function_ptr& f : prerequisites_) {
            if (!f->is_constant()) {
                return false;
            }
        }
        return true;
    }

public:
    virtual function_ptr clone() const = 0;

    virtual bool is_constant() const { return false; }

    virtual void forward(computation_context& /*context*/) {}
    virtual void backward_x(computation_context& /*context*/) {}
    virtual void backward_w(computation_context& /*context*/) const {}
    virtual void tune(computation_context& /*context*/, const float /*learning_rate*/) {}

public:
    const function_ptr& x() const {
        return prerequisites_.at(0);
    }

    const function_ptr& y() const {
        return prerequisites_.at(1);
    }

protected:
    static const matrix& x_value(const computation_context& context) {
        return *context.input.at(0);
    }

    static const matrix& y_value(const computation_context& context) {
        return *context.input.at(1);
    }

    static matrix& x_gradient(computation_context& context) {
        return *context.input_gradients.at(0);
    }

    static matrix& y_gradient(computation_context& context) {
        return *context.input_gradients.at(1);
    }

protected:
    std::vector<function_ptr> prerequisites_;
    const size_t              output_count_;
};

class constant : public function {
public:
    explicit constant(const size_t output_count) : function(0, output_count) {}

public:
    virtual function_ptr clone() const override {
        return std::make_shared<constant>(*this);
    }

    virtual bool is_constant() const override { return true; }
};

inline function_ptr make_constant(const size_t output_count) {
    return std::make_shared<constant>(output_count);
}

class column_stack : public function {
public:
    explicit column_stack(const std::vector<function_ptr>& inputs);

public:
    virtual function_ptr clone() const override {
        return std::make_shared<column_stack>(*this);
    }

    virtual void forward(computation_context& context) override;
    virtual void backward_x(computation_context& context) override;
};

inline function_ptr make_column_stack(const std::vector<function_ptr>& inputs) {
    return std::make_shared<column_stack>(inputs);
}

}
