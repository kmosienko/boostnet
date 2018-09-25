#include "tree.h"
#include "random.h"

#include <limits>

namespace boostnet {
namespace tree_machinery {

row_array compute_split_mse(const matrix& Y) {
    const uint32_t output_count = Y.cols();
    const uint32_t object_count = Y.rows();

    if (object_count == 0) {
        throw std::logic_error("no objects");
    }

    row_array y_2_mean(1, output_count);
    row_array y_mean(1, output_count);
    y_2_mean.fill(0);
    y_mean.fill(0);

    for (size_t i = 0; i < object_count; ++i) {
        y_2_mean += (Y.row(i).array().square());
        y_mean += (Y.row(i).array());
    }

    return y_2_mean - y_mean.square() / object_count;
}

size_t accurate_procent(const size_t total,
                        const scalar_type procent,
                        const size_t minimum)
{
    return std::min<size_t>(
                std::max<size_t>(procent * total, minimum),
                total);
}

std::vector<uint32_t> generate_feature_indexes(
        const size_t total,
        const size_t needed)
{
    std::vector<uint32_t> res(total, 0);
    for (size_t i = 0; i < total; ++i) {
        res[i] = i;
    }
    if (total <= needed) {
        return res;
    }
    std::random_shuffle(res.begin(), res.end());

    res.resize(needed);
    res.shrink_to_fit();
    std::sort(res.begin(), res.end());

    return res;
}

std::vector<uint32_t> generate_feature_indexes(
        const size_t total,
        const scalar_type procent,
        const size_t minimum)
{
    const size_t count = accurate_procent(total, procent, minimum);

    if (count == 0) {
        return {uint32_t(rand() % total)};
    }

    return generate_feature_indexes(total, count);
}

}

using namespace tree_machinery;

void oblivious_tree::clear() {
    this->~oblivious_tree();
    new (this) oblivious_tree;
}

void oblivious_tree::scale_values(scalar_type s) {
    for (size_t i = 0; i < values_.size(); ++i) {
        if (is_not_initialized(values_[i])) {
            continue;
        }

        values_[i] *= s;
        shrinked_values_[i] *= s;
    }
}

void oblivious_tree::scale_thresholds(const row_array& s) {
    for (size_t i = 0; i < features_.size(); ++i) {
        thresholds_[i] *= s[features_[i]];
    }
}

void oblivious_tree::shift_thresholds(const row_array& s) {
    for (size_t i = 0; i < features_.size(); ++i) {
        thresholds_[i] += s[features_[i]];
    }
}

bool oblivious_tree::grow(size_t depth,
                          bool reuse_features,
                          const matrix& X,
                          const matrix& Y,
                          const std::vector<uint32_t>& used_input_features,
                          const std::vector<uint32_t>& used_output_features,
                          uint32_t leaf_regularization)
{
    if (X.rows() != Y.rows()) {
        throw std::logic_error("bad data");
    }
    if (X.rows() == 0) {
        throw std::logic_error("empty data");
    }

    clear();

    used_output_features_ = used_output_features;
    const matrix Y_subspace = select_cols(Y, used_output_features);

    get_split_indexes(X);

    std::vector<std::vector<uint32_t>> sorting_orders(
                used_input_features.size(),
                std::vector<uint32_t>(X.rows(), 0));

    bool x_is_ok = false;
    for (size_t i = 0; i < used_input_features.size(); ++i) {
        for (size_t j = 0; j < (size_t)X.rows(); ++j) {
            sorting_orders[i][j] = j;
        }
        std::sort(sorting_orders[i].begin(), sorting_orders[i].end(),
                  [&X, &used_input_features, i](const uint32_t left, const uint32_t right) {
            return X(left, used_input_features[i]) < X(right, used_input_features[i]);
        });

        if (X(sorting_orders[i].front(), used_input_features[i]) !=
            X(sorting_orders[i].back(), used_input_features[i])) {
            x_is_ok = true;
        } else {
            sorting_orders[i].clear();
        }
    }
    if (!x_is_ok) {
        return false;
    }


    array y_sum_front;
    array y_2_sum_front;
    array y_sum_back;
    array y_2_sum_back;

    array losses;
    row_array accumulated_losses;
    std::vector<size_t> split_sizes_front;
    std::vector<size_t> split_sizes_back;

    const size_t current_depth = features_.size();
    while (depth > 0) {
        --depth;

        const size_t split_count = (1 << features_.size());

        scalar_type best_loss = std::numeric_limits<scalar_type>::max();
        size_t      best_feature = 0;
        scalar_type best_threshold = 0;

        for (size_t i = 0; i < used_input_features.size(); ++i) {
            if (!reuse_features &&
                std::find(
                        features_.begin(),
                        features_.end(),
                        used_input_features[i])
                    != features_.end())
            {
                continue;
            }

            const std::vector<uint32_t>& sorting_order = sorting_orders[i];
            if (sorting_order.empty()) {
                continue;
            }

            y_sum_front.setZero(split_count, Y_subspace.cols());
            y_2_sum_front.setZero(split_count, Y_subspace.cols());
            y_sum_back.setZero(split_count, Y_subspace.cols());
            y_2_sum_back.setZero(split_count, Y_subspace.cols());

            losses.setConstant(
                        sorting_order.size() + 1,
                        split_count,
                        std::numeric_limits<scalar_type>::max());
            split_sizes_front.clear();
            split_sizes_back.clear();
            split_sizes_front.resize(split_count, 0);
            split_sizes_back.resize(split_count, 0);

            for (size_t j = 1; j < sorting_order.size() + 1; ++j) {
                const uint32_t object_index_front = sorting_order[j - 1];
                const uint32_t split_index_front = split_indexes_(object_index_front);

                const auto& y_front = Y_subspace.row(object_index_front);

                ++split_sizes_front[split_index_front];
                const scalar_type size = split_sizes_front[split_index_front];

                y_sum_front.row(split_index_front) += y_front.array();
                y_2_sum_front.row(split_index_front) += y_front.array().square();

                const scalar_type loss_front =
                        (y_2_sum_front.row(split_index_front) -
                         y_sum_front.row(split_index_front).square() / size).sum();

                if (losses(j, split_index_front) == std::numeric_limits<scalar_type>::max()) {
                    losses(j, split_index_front) = 0;
                }
                losses(j, split_index_front) += loss_front;
            }

            for (size_t j = sorting_order.size(); j > 0; --j) {
                const uint32_t object_index_back = sorting_order[j - 1];
                const uint32_t split_index_back = split_indexes_(object_index_back);

                const auto& y_back = Y_subspace.row(object_index_back);

                ++split_sizes_back[split_index_back];
                const scalar_type size = split_sizes_back[split_index_back];

                y_sum_back.row(split_index_back) += y_back.array();
                y_2_sum_back.row(split_index_back) += y_back.array().square();

                const scalar_type loss_back =
                        (y_2_sum_back.row(split_index_back) -
                         y_sum_back.row(split_index_back).square() / size).sum();

                if (losses(j, split_index_back) == std::numeric_limits<scalar_type>::max()) {
                    losses(j, split_index_back) = 0;
                }
                losses(j, split_index_back) += loss_back;
            }

            for (size_t j = 0; j < split_count; ++j) {
                if (split_sizes_front[j] == 0) {
                    losses(0, j) = 0;
                } else {
                    const scalar_type size = split_sizes_front[j];

                    losses(0, j) =
                            (y_2_sum_front.row(j) -
                             y_sum_front.row(j).square() / size).sum();
                }
            }

            accumulated_losses = losses.row(0);

            for (size_t j = 1; j < sorting_order.size(); ++j) {
                for (size_t k = 0; k < split_count; ++k) {
                    if (losses(j, k) != std::numeric_limits<scalar_type>::max()) {
                        accumulated_losses(k) = losses(j, k);
                    }
                }

                const scalar_type loss = accumulated_losses.sum();

                if (loss < best_loss) {
                    best_loss = loss;
                    best_feature = used_input_features[i];
                    best_threshold = X(sorting_order[j], best_feature);
                    for (size_t k = j; k > 0; --k) {
                        if (X(sorting_order[j], best_feature) != X(sorting_order[k - 1], best_feature)) {
                            best_threshold =
                                    (X(sorting_order[j], best_feature) +
                                     X(sorting_order[k - 1], best_feature)) / 2.0;
                            break;
                        }
                    }
                }
            }
        }

        if (best_loss == std::numeric_limits<scalar_type>::max()) {
            break;
        }
        if (!features_.empty() &&
            features_.back() == best_feature &&
            thresholds_.back() == best_threshold)
        {
            break;
        }

        features_.push_back(best_feature);
        thresholds_.push_back(best_threshold);
        values_.resize(split_count * 2);
        shrinked_values_.resize(split_count * 2);

        get_split_indexes(X);
    }

    if (current_depth == features_.size()) {
        return false;
    }

    compute_split_values(
                used_output_features,
                Y_subspace,
                Y.cols(),
                leaf_regularization);

    fill_split_properties(Y.cols());

    return true;
}

void oblivious_tree::tune(const matrix& X,
                          const matrix& out_gradient,
                          scalar_type learning_rate,
                          const bool tune_values,
                          const bool tune_thresholds)
{
    const size_t batch_size = split_indexes_.rows();
    const size_t split_count = (1 << features_.size());

    const array used_out_gradient = select_cols(out_gradient, used_output_features_);

    if (tune_values) {
        if (is_not_initialized(split_value_gradient_)) {
            split_value_gradient_.setZero(split_count, used_output_features_.size());
        }

        array split_value_gradient;
        split_value_gradient.setZero(split_count, used_output_features_.size());

        for (size_t i = 0; i < batch_size; ++i) {
//            for (uint32_t split_index : possible_splits_) {
//                split_value_gradient.row(split_index) +=
//                        path_coefficients_(i, split_index) * used_out_gradient.row(i);
//            }
            split_value_gradient.row(split_indexes_[i]) += used_out_gradient.row(i);
        }

        split_value_gradient_ =
                0.8 * split_value_gradient_ +
                0.2 * split_value_gradient;

        for (uint32_t split_index : possible_splits_) {
            shrinked_values_[split_index] -= learning_rate * split_value_gradient_.row(split_index).matrix();
            for (size_t j = 0; j < used_output_features_.size(); ++j) {
                values_[split_index][used_output_features_[j]] = shrinked_values_[split_index][j];
            }
        }
    }
    if (tune_thresholds) {
        for (size_t i = 0; i < thresholds_.size(); ++i) {
            thresholds_[i] -= learning_rate * split_thresholds_gradient_[i];
        }

        fill_split_properties(out_gradient.cols());
    }
}

void oblivious_tree::init_random(const size_t depth,
                                 const size_t feature_count,
                                 const std::vector<uint32_t>& used_output_features,
                                 const size_t output_count,
                                 const row_array& col_means,
                                 const row_array& col_stddevs)
{
    clear();

    used_output_features_ = used_output_features;

    features_ = generate_feature_indexes(feature_count, depth);
    thresholds_.resize(features_.size());

    for (size_t i = 0; i < thresholds_.size(); ++i) {
        thresholds_[i] = normal_scalar(
                    col_means[features_[i]],
                    2 * col_stddevs[features_[i]]);
    }

    const size_t split_count = (1 << features_.size());

    shrinked_values_.resize(split_count);
    values_.resize(split_count);

    for (size_t i = 0; i < split_count; ++i) {
        shrinked_values_[i].setZero(used_output_features_.size());
        values_[i].setZero(output_count);

        fill_normal(shrinked_values_[i]);
        for (size_t j = 0; j < used_output_features_.size(); ++j) {
            values_[i][used_output_features_[j]] = shrinked_values_[i][j];
        }
    }

    fill_split_properties(output_count);
}

void oblivious_tree::compute_split_values(
        const std::vector<uint32_t>& used_output_features,
        const matrix& Y_subspace,
        const size_t output_count,
        uint32_t leaf_regularization)
{
    const size_t split_count = (1 << this->features_.size());

    std::vector<uint32_t> split_sizes(split_count, 0);
    for (Eigen::Index i = 0; i < split_indexes_.rows(); ++i) {
        ++split_sizes.at(split_indexes_[i]);
    }

    std::vector<row_vector> sum_Y;
    sum_Y.resize(split_count, row_vector::Zero(used_output_features.size()));

    const size_t batch_size = split_indexes_.rows();

    for (size_t i = 0; i < batch_size; ++i) {
        const uint32_t split_index = split_indexes_[i];
        row_vector& y = sum_Y.at(split_index);

        y += Y_subspace.row(i);
    }

    for (size_t i = 0; i < split_count; ++i) {
        values_[i] = row_vector();
        shrinked_values_[i] = row_vector();
        if (split_sizes.at(i) == 0) {
            continue;
        }

        scalar_type fsize = (scalar_type)split_sizes.at(i);
        fsize *= sqrt((fsize + leaf_regularization) / fsize);

        values_[i].setZero(output_count);
        for (size_t j = 0; j < used_output_features_.size(); ++j) {
            values_[i](used_output_features_[j]) = sum_Y[i](j) / fsize;
        }

        shrinked_values_[i] = sum_Y[i] / fsize;
    }
}

void oblivious_tree::gradient(const matrix& features,
                              const matrix& out_gradient,
                              matrix& gradient) const
{
    fill_sigmoids(features);

    const array used_out_gradient = select_cols(out_gradient, used_output_features_);
    const size_t object_count = (size_t)features.rows();

    column_array sigmoid_sums;
    std::vector<bool> used_features;

    column_array split_thresholds_gradient;
    split_thresholds_gradient.setZero(thresholds_.size());

    for (size_t i = 0; i < object_count; ++i) {
        for (const uint32_t split_index : possible_splits_) {
            scalar_type sigmoid_product = 1;
            sigmoid_sums.setZero(features_.size());
            for (size_t j = 0; j < features_.size(); ++j) {
                if (!used_comparisons_[split_index][j]) {
                    continue;
                }

                if (split_index & (1 << (features_.size() - j - 1))) {
                    sigmoid_product *= sigmoids_(i, j);
                    sigmoid_sums[feature_indexes_[j]] += (1 - sigmoids_(i, j));
                } else {
                    sigmoid_product *= (1 - sigmoids_(i, j));
                    sigmoid_sums[feature_indexes_[j]] -= sigmoids_(i, j);
                }
            }

            used_features.clear();
            used_features.resize(features_.size(), false);

            for (size_t j = 0; j < features_.size(); ++j) {
                const scalar_type grad_prod =
                        (used_out_gradient.row(i) * shrinked_values_[split_index].array()).sum();

                if (split_index & (1 << (features_.size() - j - 1))) {
                    split_thresholds_gradient[j] -= sigmoid_product * (1 - sigmoids_(i, j)) * grad_prod;
                } else {
                    split_thresholds_gradient[j] += sigmoid_product * sigmoids_(i, j) * grad_prod;
                }

                if (used_features[feature_indexes_[j]]) {
                    continue;
                }

                gradient(i, features_[j]) +=
                        grad_prod * sigmoid_product * sigmoid_sums[feature_indexes_[j]];

                used_features[feature_indexes_[j]] = true;
            }
        }
    }

    if (is_not_initialized(split_thresholds_gradient_)) {
        split_thresholds_gradient_.setZero(thresholds_.size());
    }

    split_thresholds_gradient_ =
            0.8 * split_thresholds_gradient_ +
            0.2 * split_thresholds_gradient;
}

void oblivious_tree::gradient_fast(const matrix& features,
                                   const matrix& out_gradient,
                                   matrix& gradient) const
{
    fill_sigmoids(features);

    const array used_out_gradient = select_cols(out_gradient, used_output_features_);
    const size_t object_count = (size_t)features.rows();

    column_array split_thresholds_gradient;
    split_thresholds_gradient.setZero(thresholds_.size());

    for (size_t i = 0; i < object_count; ++i) {
        const uint32_t split_index = split_indexes_(i);

        const row_vector& this_value = shrinked_values_[split_index];
        if (is_not_initialized(this_value)) {
            continue;
        }

        for (size_t j = 0; j < features_.size(); ++j) {
            const uint32_t power_of_two = (1 << (features_.size() - j - 1));

            const uint32_t path_direction = split_index & power_of_two;
            const uint32_t other_split =
                    path_direction != 0 ?
                        (split_index - power_of_two) :
                        (split_index + power_of_two);
            const row_vector& other_value = shrinked_values_[other_split];
            if (is_not_initialized(other_value)) {
                continue;
            }

            const scalar_type grad =
                    sigmoids_(i, j) * (1 - sigmoids_(i, j)) *
                    (used_out_gradient.row(i) *
                     (this_value.array() - other_value.array()))
                    .sum();

            if (path_direction != 0) {
                split_thresholds_gradient[j] -= grad;
                gradient(i, features_[j]) += grad;
            } else {
                split_thresholds_gradient[j] += grad;
                gradient(i, features_[j]) -= grad;
            }
        }
    }

    if (is_not_initialized(split_thresholds_gradient_)) {
        split_thresholds_gradient_.setZero(thresholds_.size());
    }

    split_thresholds_gradient_ =
            0.8 * split_thresholds_gradient_ +
            0.2 * split_thresholds_gradient;
}

scalar_type oblivious_tree::compute_entropy() const {
    const size_t batch_size = split_indexes_.rows();
    const size_t split_count = (1 << this->features_.size());

    std::vector<uint32_t> split_sizes(split_count, 0);
    for (size_t i = 0; i < batch_size; ++i) {
        if (!is_not_initialized(shrinked_values_[split_indexes_[i]])) {
            ++split_sizes.at(split_indexes_[i]);
        }
    }

    scalar_type res = 0;
    for (size_t i = 0; i < split_count; ++i) {
        if (split_sizes[i] == 0) {
            continue;
        }

        const scalar_type prob =
                ((scalar_type)split_sizes[i]) / ((scalar_type)batch_size);

        res -= prob * log(prob);
    }
    return res;
}

void oblivious_tree::get_split_indexes(const matrix& features) {
    split_indexes_.setZero(features.rows());

    if (features_.empty()) {
        return;
    }

    uint32_t power_of_two = 1 << (features_.size() - 1);

    for (size_t i = 0; i < features_.size(); ++i) {
        if (power_of_two == 0) {
            throw std::logic_error("bad algorithm");
        }

        split_indexes_ += power_of_two * (features.col(features_[i]).array() > thresholds_[i]).cast<uint32_t>();

        power_of_two /= 2;
    }

    if (power_of_two != 0) {
        throw std::logic_error("bad algorithm");
    }
}

void oblivious_tree::fill_sigmoids(const matrix& features) const {
    sigmoids_.setZero(features.rows(), features_.size());

    for (size_t i = 0; i < features_.size(); ++i) {
        sigmoids_.col(i) = thresholds_[i] - features.col(features_[i]).array();
    }

    sigmoids_ = Eigen::exp(sigmoids_);
    sigmoids_ += 1;
    sigmoids_ = Eigen::inverse(sigmoids_);
}

void oblivious_tree::fill_sigmoids_fast(const matrix& features) const {
    const size_t batch_size = features.rows();

    sigmoids_.setZero(features.rows(), features_.size());

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < features_.size(); ++j) {
            scalar_type x = features(i, features_[j]) - thresholds_[j];
            bool positive = (x >= 0);
            x = fabs(x);

            if (x >= 5) {
                sigmoids_(i, j) = 1;
            } else if (x >= 2.375) {
                sigmoids_(i, j) = 0.03125 * x + 0.84375;
            } else if (x >= 1) {
                sigmoids_(i, j) = 0.125 * x + 0.625;
            } else {
                sigmoids_(i, j) = 0.25 * x + 0.5;
            }

            if (!positive) {
                sigmoids_(i, j) = 1.0 - sigmoids_(i, j);
            }
        }
    }
}

void oblivious_tree::fill_path_coefficients(
        const matrix& sigmoids,
        array& coefficients) const
{
    coefficients.setZero(sigmoids.rows(), values_.size());

    std::vector<column_array> tmp1;
    std::vector<column_array> tmp2;
    tmp1.reserve(values_.size());
    tmp2.reserve(values_.size());

    tmp1 = {1.f - sigmoids.col(0).array(), sigmoids.col(0)};

    std::vector<column_array>* current = &tmp1;
    std::vector<column_array>* next = &tmp2;

    for (Eigen::Index i = 1; i < sigmoids.cols(); ++i) {
        next->resize(current->size() * 2);
        for (size_t j = 0; j < current->size(); ++j) {
            next->operator [](j * 2)     = current->operator [](j) * (1.f - sigmoids.col(i).array());
            next->operator [](j * 2 + 1) = current->operator [](j) *        sigmoids.col(i).array();
        }

        std::swap(current, next);
    }

    if (coefficients.cols() != (Eigen::Index)current->size()) {
        throw std::logic_error("bad algorithm");
    }

    for (size_t i = 0; i < current->size(); ++i) {
        coefficients.col(i) = current->operator [](i);
    }
}

void oblivious_tree::fill_split_properties(const size_t output_count) {
    feature_indexes_.clear();
    size_t used_feature_count = 0;
    for (size_t i = 0; i < features_.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (features_[i] == features_[j]) {
                feature_indexes_.push_back(feature_indexes_[j]);
                break;
            }
        }
        if (feature_indexes_.size() != i + 1) {
            feature_indexes_.push_back(used_feature_count++);
        }
    }

    BN_VERIFY(feature_indexes_.size() == features_.size());

    const size_t split_count = shrinked_values_.size();

    possible_splits_.clear();
    used_comparisons_.clear();
    used_comparisons_.resize(
                split_count,
                std::vector<bool>(features_.size(), true));

    std::vector<std::pair<scalar_type, scalar_type>> possible_ranges;
    for (size_t i = 0; i < split_count; ++i) {
        possible_ranges.clear();
        possible_ranges.resize(
                    used_feature_count,
                    std::make_pair(
                        std::numeric_limits<scalar_type>::lowest(),
                        std::numeric_limits<scalar_type>::max()));

        std::vector<bool>& used_comparisons = used_comparisons_[i];

        bool bad_split = false;
        const uint32_t split_index = i;
        for (size_t j = 0; j < features_.size(); ++j) {
            const uint32_t path_goes_to_the_right = split_index & (1 << (features_.size() - j - 1));

            std::pair<scalar_type, scalar_type>& range = possible_ranges[feature_indexes_[j]];
            const scalar_type t = thresholds_[j];

            if (t <= range.first && path_goes_to_the_right) {
                used_comparisons[j] = false;
            } else if (t >= range.second && !path_goes_to_the_right) {
                used_comparisons[j] = false;
            } else if (!(t > range.first && t < range.second)) {
                bad_split = true;
                break;
            }
            if (path_goes_to_the_right) {
                range.first = std::max(range.first, t);
            } else {
                range.second = std::min(range.second, t);
            }
        }

        if (!bad_split) {
            possible_splits_.push_back(i);

            if (is_not_initialized(shrinked_values_[i])) {
                shrinked_values_[i].setZero(used_output_features_.size());
                values_[i].setZero(output_count);
            }
        } else {
            used_comparisons.clear();

            if (!is_not_initialized(shrinked_values_[i])) {
                shrinked_values_[i] = row_vector();
                values_[i] = row_vector();
            }
        }
    }
}

}
