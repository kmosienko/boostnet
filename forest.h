#pragma once

#include "function.h"
#include "tree.h"
#include "random.h"

#include <list>
#include <deque>

namespace boostnet {

struct forest_config {
    enum class x_selection_method {
        random
    };

    enum class y_selection_method {
        random,
        max_mse,
        max_square
    };

    size_t             tree_depth          = 5;
    x_selection_method x_selection         = x_selection_method::random;
    y_selection_method y_selection         = y_selection_method::max_square;
    float              x_subsample         = 0.2;
    float              y_subsample         = 0.2;
    size_t             x_minimum           = 20;
    size_t             y_minimum           = 20;
    size_t             max_trees           = 1000;
    size_t             min_trees           = 10;
    uint32_t           leaf_regularization = 0;
    bool               reuse_features      = true;
    size_t             minibatch_for_tune  = 10;
    float              dropout             = 0.5;

    forest_config set_tree_depth(size_t tree_depth_) {
        forest_config new_config = *this;
        new_config.tree_depth = tree_depth_;
        return new_config;
    }

    forest_config set_x_selection(x_selection_method x_selection_) {
        forest_config new_config = *this;
        new_config.x_selection = x_selection_;
        return new_config;
    }

    forest_config set_y_selection(y_selection_method y_selection_) {
        forest_config new_config = *this;
        new_config.y_selection = y_selection_;
        return new_config;
    }

    forest_config set_x_subsample(float x_subsample_) {
        forest_config new_config = *this;
        new_config.x_subsample = x_subsample_;
        return new_config;
    }

    forest_config set_y_subsample(float y_subsample_) {
        forest_config new_config = *this;
        new_config.y_subsample = y_subsample_;
        return new_config;
    }

    forest_config set_x_minimum(size_t x_minimum_) {
        forest_config new_config = *this;
        new_config.x_minimum = x_minimum_;
        return new_config;
    }

    forest_config set_y_minimum(size_t y_minimum_) {
        forest_config new_config = *this;
        new_config.y_minimum = y_minimum_;
        return new_config;
    }

    forest_config set_trees(size_t trees_) {
        forest_config new_config = *this;
        new_config.max_trees = trees_;
        new_config.min_trees = trees_;
        return new_config;
    }

    forest_config set_max_trees(size_t max_trees_) {
        forest_config new_config = *this;
        new_config.max_trees = max_trees_;
        return new_config;
    }

    forest_config set_min_trees(size_t min_trees_) {
        forest_config new_config = *this;
        new_config.min_trees = min_trees_;
        return new_config;
    }

    forest_config set_leaf_regularization(uint32_t leaf_regularization_) {
        forest_config new_config = *this;
        new_config.leaf_regularization = leaf_regularization_;
        return new_config;
    }

    forest_config set_reuse_features(bool reuse_features_) {
        forest_config new_config = *this;
        new_config.reuse_features = reuse_features_;
        return new_config;
    }

    forest_config set_minibatch_for_tune(size_t minibatch_for_tune_) {
        forest_config new_config = *this;
        new_config.minibatch_for_tune = minibatch_for_tune_;
        return new_config;
    }

    forest_config set_dropout(float dropout_) {
        forest_config new_config = *this;
        new_config.dropout = dropout_;
        return new_config;
    }
};

template<typename T>
class forest : public function {
public:
    forest(function_ptr x,
           const size_t output_count,
           const forest_config& config = forest_config())
        : function(1, output_count)
        , config_(config)
    {
        prerequisites_.at(0) = x;
    }

public:
    virtual function_ptr clone() const override {
        return std::make_shared<forest>(*this);
    }

    virtual void forward(computation_context& context) override {
        const matrix& input = x_value(context);
        context.output.setZero(input.rows(), get_output_count());

        if (trees_.empty()) {
            const row_array col_means = tree_machinery::compute_col_means(input);
            const row_array col_stddevs = tree_machinery::compute_col_stddevs(input, col_means);

            if ((col_stddevs == 0).all()) {
                return;
            }

            while (trees_.size() < config_.min_trees) {
                trees_.emplace_back();
                tree_holder& t = trees_.back();

                while (t.entropy == 0) {
                    const std::vector<uint32_t> used_output_features =
                            tree_machinery::generate_feature_indexes(
                                get_output_count(),
                                config_.y_subsample,
                                config_.y_minimum);

                    t.tree.init_random(
                                config_.tree_depth,
                                input.cols(),
                                used_output_features,
                                get_output_count(),
                                col_means,
                                col_stddevs);

                    t.tree.scale_values(1.0 / config_.min_trees);

                    t.tree.get_split_indexes(input);
                    t.entropy = t.tree.compute_entropy();
                }
            }
        }

        for (tree_holder& t : trees_) {
            t.gradient_computed = false;
            t.selected_for_tune =
                    config_.dropout == 0 ?
                        true :
                        (uniform_scalar() > config_.dropout);

            t.tree.call(input, context.output);
        }
    }

    virtual void backward_x(computation_context& context) override {
        for (tree_holder& t : trees_) {
            if (!t.selected_for_tune) {
                continue;
            }
            t.gradient_computed = true;

            t.tree.gradient_fast(x_value(context), context.output_gradient, x_gradient(context));
        }
    }

    virtual void tune(computation_context& context, const float learning_rate) override {
        const matrix& input = x_value(context);

        matrix target = -context.output_gradient * learning_rate;

        std::vector<uint32_t> used_input_features =
                tree_machinery::generate_feature_indexes(
                    x_value(context).cols(),
                    config_.x_subsample,
                    config_.x_minimum);

        std::vector<uint32_t> used_output_features = [&]() {
            switch (config_.y_selection) {
            case forest_config::y_selection_method::random:
                return tree_machinery::generate_feature_indexes(
                            target.cols(),
                            config_.y_subsample,
                            config_.y_minimum);
            case forest_config::y_selection_method::max_mse:
            case forest_config::y_selection_method::max_square:
                const row_array target_mse =
                        (config_.y_selection == forest_config::y_selection_method::max_square) ?
                            target.array().square().colwise().mean() :
                            tree_machinery::compute_split_mse(target);

                std::vector<uint32_t> used_output_features;
                for (Eigen::Index i = 0; i < target.cols(); ++i) {
                    used_output_features.push_back(i);
                }

                std::sort(used_output_features.begin(), used_output_features.end(), [&target_mse](uint32_t l, uint32_t r) {
                    return target_mse(l) > target_mse(r);
                });

                used_output_features.resize(
                            tree_machinery::accurate_procent(
                                used_output_features.size(),
                                config_.y_subsample,
                                config_.y_minimum));
                std::sort(used_output_features.begin(), used_output_features.end());

                return used_output_features;
            }
            throw std::logic_error("unknown y_selection");
        }();

        size_t deleted_count = 0;
        matrix tmp_output;
        matrix tmp_input_grad;
        {
            std::vector<decltype(trees_.begin())> to_delete;
            std::vector<decltype(trees_.begin())> to_smart_delete;
            tmp_input_grad.setZero(input.rows(), input.cols());
            for (auto it = trees_.begin(); it != trees_.end(); ++it) {
                tree_holder& t = *it;

                if (!t.selected_for_tune) {
                    continue;
                }

                tmp_output.setZero(input.rows(), get_output_count());
                t.tree.call(input, tmp_output);

                t.derivative =
                        0.8 * t.derivative +
                        0.2 * (tmp_output.array() * context.output_gradient.array()).sum() / t.contribution;
                t.derivative = std::clamp<scalar_type>(t.derivative, -10, 10);

                const scalar_type old_contribution = t.contribution;
                t.contribution -= learning_rate * t.derivative;
                const scalar_type scale = t.contribution / old_contribution;

                if (t.contribution <= 0 /*|| (!(scale >= 10 && scale <= 10))*/) {
                    to_delete.push_back(it);

                    continue;
                }

                if (t.tree.compute_entropy() <= 0.0 * t.entropy) {
                    to_smart_delete.push_back(it);
                }

                t.tree.scale_values(scale);
                if (scale < 0) {
                    t.contribution = -t.contribution;
                    t.derivative = -t.derivative;
                }

                if (!t.gradient_computed) {
                    t.tree.gradient_fast(input, context.output_gradient, tmp_input_grad);
                    t.gradient_computed = true;
                }

                t.tree.tune(input,
                            context.output_gradient,
                            learning_rate,
                            true,
                            true);
            }

            for (const auto it : to_delete) {
                trees_.erase(it);

                ++deleted_count; //std::cerr << 'c' << std::endl;
            }
            for (auto it : to_smart_delete) {
                tree_holder& t = *it;

                for (const uint32_t f : t.tree.get_split_features()) {
                    used_input_features.push_back(f);
                }
                for (const uint32_t f : t.tree.get_used_output_features()) {
                    used_output_features.push_back(f);
                }

                t.tree.call(input, target);
                trees_.erase(it);

                ++deleted_count; //std::cerr << 'e' << std::endl;
            }

            if (deleted_count > 0) {
                std::sort(used_input_features.begin(), used_input_features.end());
                std::sort(used_output_features.begin(), used_output_features.end());

                used_input_features.erase(
                            std::unique(
                                used_input_features.begin(),
                                used_input_features.end()),
                            used_input_features.end());
                used_output_features.erase(
                            std::unique(
                                used_output_features.begin(),
                                used_output_features.end()),
                            used_output_features.end());
            }
        }

        for (size_t i = 0; i <= deleted_count; ++i) {
            if (trees_.size() >= config_.max_trees) {
                break;
            }

            trees_.emplace_back();
            tree_holder& t = trees_.back();

            t.tree.grow(config_.tree_depth,
                        config_.reuse_features,
                        x_value(context),
                        target,
                        used_input_features,
                        used_output_features,
                        config_.leaf_regularization);

            if (trees_.back().tree.get_split_features().empty()) {
                trees_.pop_back();
                break;
            } else {
                t.tree.template call<-1>(input, target);
                t.entropy = t.tree.compute_entropy();
            }
        }
    }

private:
    struct tree_holder {
        T           tree;
        scalar_type entropy;
        scalar_type contribution;
        scalar_type derivative;
        bool        selected_for_tune;
        bool        gradient_computed;

        tree_holder()
            : tree()
            , entropy(0)
            , contribution(1)
            , derivative(0)
            , selected_for_tune(true)
            , gradient_computed(false)
        {}
    };

    forest_config           config_;
    std::list<tree_holder>  trees_;
};

template<typename T>
inline function_ptr make_forest(
        function_ptr x,
        size_t output_count,
        const forest_config& config = forest_config())
{
    return std::make_shared<forest<T>>(x, output_count, config);
}

template<typename T>
function_ptr make_forest_array(
        function_ptr x,
        size_t output_count,
        const forest_config& config = forest_config())
{
    std::vector<function_ptr> forests;
    forests.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        forests.push_back(make_forest<T>(x, 1, config));
    }
    return make_column_stack(forests);
}

}
