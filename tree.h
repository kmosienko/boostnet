#pragma once

#include "function.h"

#include <inttypes.h>

#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <unordered_set>

namespace boostnet {
namespace tree_machinery {

struct split_loss_data {
    uint32_t feature_;
    scalar_type    threshold_;
    scalar_type    error_;
    array    y_left_;
    array    y_right_;

    split_loss_data()
        : feature_(-1)
        , threshold_(0)
        , error_(0)
        , y_left_()
        , y_right_()
    {}
};

row_array compute_split_mse(const matrix& Y);

size_t accurate_procent(const size_t total,
                        const scalar_type procent,
                        const size_t minimum);

std::vector<uint32_t> generate_feature_indexes(
        const size_t total,
        const size_t needed);

std::vector<uint32_t> generate_feature_indexes(
        const size_t total,
        const scalar_type procent = 100500,
        const size_t minimum = 0);

template<typename T, typename U = T>
U select_cols(
        const T& m,
        const std::vector<uint32_t>& c)
{
    U res(m.rows(), c.size());

    for (size_t i = 0; i < c.size(); ++i) {
        res.col(i) = m.col(c[i]);
    }

    return res;
}

template<typename T, typename U = T>
U select_rows(
        const T& m,
        const std::vector<uint32_t>& r) {
    U res(r.size(), m.cols());

    for (size_t i = 0; i < r.size(); ++i) {
        res.row(i) = m.row(r.at(i));
    }

    return res;
}

template<typename T>
row_array compute_col_means(const T& Y) {
    return Y.colwise().mean().array();
}

template<typename T, typename U>
row_array compute_col_stddevs(const T& Y, const U& col_means) {
    const array y = Y;
    const row_array m = col_means;
    const array r = y.rowwise() - m;
    return compute_col_means(r.square()).sqrt();
}

template<typename T>
row_array compute_col_stddevs(const T& Y) {
    return compute_col_stddevs(Y, compute_col_means(Y));
}

}

class oblivious_tree {
public:
    void clear();

    template<int32_t coefficient = 1>
    void call(
            const matrix& features,
            matrix& result)
    {
        if (features_.empty()) {
            return;
        }

        get_split_indexes(features);

        for (Eigen::Index i = 0; i < features.rows(); ++i) {
            uint32_t split = split_indexes_[i];
            const row_vector& y = shrinked_values_.at(split);

            if (is_not_initialized(y)) {
                continue;
            }

            for (size_t j = 0; j < used_output_features_.size(); ++j) {
                if (coefficient == 1) {
                    result(i, used_output_features_[j]) += y[j];
                } else if (coefficient == -1) {
                    result(i, used_output_features_[j]) -= y[j];
                } else {
                    result(i, used_output_features_[j]) += coefficient * y[j];
                }
            }
        }
    }

    void scale_values(scalar_type s);
    void scale_thresholds(const row_array& s);
    void shift_thresholds(const row_array& s);

    bool grow(size_t depth,
              bool reuse_features,
              const matrix& X,
              const matrix& Y,
              const std::vector<uint32_t>& used_input_features,
              const std::vector<uint32_t>& used_output_features,
              uint32_t leaf_regularization = 0);

    void tune(const matrix& X,
              const matrix& out_gradient,
              scalar_type learning_rate,
              const bool tune_values = true,
              const bool tune_thresholds = true);

    void init_random(const size_t depth,
                     const size_t feature_count,
                     const std::vector<uint32_t>& used_output_features,
                     const size_t output_count,
                     const row_array& col_means,
                     const row_array& col_stddevs);

    const std::vector<uint32_t>&    get_split_features()       const { return features_; }
    const std::vector<scalar_type>& get_split_thresholds()     const { return thresholds_; }
    const std::vector<row_vector>&  get_split_values()         const { return values_; }
    const std::vector<uint32_t>&    get_used_output_features() const { return used_output_features_; }

    void gradient(const matrix& features,
                  const matrix& out_gradient,
                  matrix& gradient) const;
    void gradient_fast(const matrix& features,
                       const matrix& out_gradient,
                       matrix& gradient) const;

    void get_split_indexes(const matrix& features);
    scalar_type compute_entropy() const;

private:
    void compute_split_values(
            const std::vector<uint32_t>& used_output_features,
            const matrix& Y_subspace,
            const size_t output_count,
            uint32_t leaf_regularization);

    void fill_sigmoids(const matrix& features) const;
    void fill_sigmoids_fast(const matrix& features) const;

    void fill_path_coefficients(
            const matrix& sigmoids,
            array& coefficients) const;

    void fill_split_properties(const size_t output_count);

private:
    // tree structure
    std::vector<uint32_t>    features_;
    std::vector<scalar_type> thresholds_;
    std::vector<row_vector>  values_;
    std::vector<row_vector>  shrinked_values_;
    std::vector<uint32_t>    used_output_features_;

    // split properties
    std::vector<std::vector<bool>> used_comparisons_;
    std::vector<uint32_t>          possible_splits_;
    std::vector<size_t>            feature_indexes_;

    // batch tmp variables
    mutable column_array_t<uint32_t> split_indexes_;
    mutable array                    sigmoids_;
    mutable array                    split_value_gradient_;
    mutable column_array             split_thresholds_gradient_;
};

}
