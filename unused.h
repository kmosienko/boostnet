
//void oblivious_tree::gradient(const matrix& features,
//                              const matrix& out_gradient,
//                              matrix& gradient) const
//{
//    fill_sigmoids(features);

//    array path_coefficients;
//    fill_path_coefficients(sigmoids_, path_coefficients);

//    const array used_out_gradient = select_cols(out_gradient, used_output_features_);

//    array tmp;
//    for (size_t j = 0; j < features_.size(); ++j) {
//        tmp.setZero(features.rows(), used_output_features_.size());

//        for (size_t k = 0; k < values_.size(); ++k) {
//            if (is_not_initialized(values_[k])) {
//                continue;
//            }

//            const size_t path_direction = k & (1 << (features_.size() - j - 1));

//            if (path_direction != 0) {
//                tmp += ((path_coefficients.col(k) * (scalar_type(1) - sigmoids_.col(j))).matrix() * shrinked_values_[k]).array();
//            } else {
//                tmp += ((path_coefficients.col(k) * (-sigmoids_.col(j))).matrix() * shrinked_values_[k]).array();
//            }
//        }

//        gradient.col(features_[j]) += ((tmp * used_out_gradient).rowwise().sum()).matrix();
//    }
//}

//void oblivious_tree::gradient2(const matrix& features,
//                               const matrix& out_gradient,
//                               matrix& gradient) const
//{
//    fill_sigmoids(features);

//    const array used_out_gradient = select_cols(out_gradient, used_output_features_);

//    const size_t object_count = (size_t)features.rows();

//    const row_vector zero = row_vector::Zero(used_output_features_.size());

//    for (size_t i = 0; i < object_count; ++i) {
//        const uint32_t split_index = split_indexes_(i);

////        if (is_not_initialized(shrinked_values_[split_index])) {
////            continue;
////        }

//        const row_vector& this_value =
//                is_not_initialized(shrinked_values_[split_index]) ?
//                    zero :
//                    shrinked_values_[split_index];

//        for (size_t j = 0; j < features_.size(); ++j) {
//            const uint32_t power_of_two = (1 << (features_.size() - j - 1));

//            const uint32_t path_direction = split_index & power_of_two;

//            if (path_direction != 0) {
//                const uint32_t other_split = split_index - power_of_two;

////                if (is_not_initialized(shrinked_values_[other_split])) {
////                    continue;
////                }

//                const row_vector& other_value =
//                        is_not_initialized(shrinked_values_[other_split]) ?
//                            zero :
//                            shrinked_values_[other_split];

//                gradient(i, features_[j]) +=
//                        sigmoids_(i, j) * (1 - sigmoids_(i, j)) *
//                        (used_out_gradient.row(i) *
//                         (this_value.array() - other_value.array()))
//                        .sum();
//            } else {
//                const uint32_t other_split = split_index + power_of_two;

////                if (is_not_initialized(shrinked_values_[other_split])) {
////                    continue;
////                }

//                const row_vector& other_value =
//                        is_not_initialized(shrinked_values_[other_split]) ?
//                            zero :
//                            shrinked_values_[other_split];

//                gradient(i, features_[j]) +=
//                        sigmoids_(i, j) * (1 - sigmoids_(i, j)) *
//                        (used_out_gradient.row(i) *
//                         (other_value.array() - this_value.array()))
//                        .sum();
//            }
//        }
//    }
//}

//void oblivious_tree::gradient3(const matrix& features,
//                               const matrix& out_gradient,
//                               matrix& gradient) const
//{
//    array sigmoids;
//    sigmoids.setZero(features.rows(), features_.size());

//    const array used_out_gradient = select_cols(out_gradient, used_output_features_);
//    const row_vector zero = row_vector::Zero(used_output_features_.size());

//    std::vector<bool>        processed(features_.size(), false);

//    std::vector<size_t>      feature_positions;
//    std::vector<scalar_type> feature_thresholds;
//    std::vector<size_t>      thresholds_sorting_order;
//    std::vector<scalar_type> sigmoid_scales;

//    for (size_t i = 0; i < features_.size(); ++i) {
//        if (processed[i]) {
//            continue;
//        }

//        feature_positions.clear();
//        feature_thresholds.clear();
//        thresholds_sorting_order.clear();
//        sigmoid_scales.clear();

//        for (size_t j = i; j < features_.size(); ++j) {
//            if (features_[j] == features_[i]) {
//                processed[j] = true;
//                feature_positions.push_back(j);
//                feature_thresholds.push_back(thresholds_[j]);
//                thresholds_sorting_order.push_back(
//                            thresholds_sorting_order.size());
//            }
//        }
//        sigmoid_scales.resize(feature_positions.size(), 1);

//        if (thresholds_sorting_order.size() > 1) {
//            std::sort(thresholds_sorting_order.begin(),
//                      thresholds_sorting_order.end(),
//                      [&feature_thresholds](const size_t l, const size_t r) {
//                return feature_thresholds[l] < feature_thresholds[r];
//            });

//            for (size_t j = 0; j < thresholds_sorting_order.size(); ++j) {
//                scalar_type scale = 0;
//                const scalar_type current_threshold =
//                        feature_thresholds[thresholds_sorting_order[j]];

//                if (j > 0) {
//                    const scalar_type prev_threshold =
//                            feature_thresholds[thresholds_sorting_order[j - 1]];

//                    BN_VERIFY(prev_threshold <= current_threshold);

//                    const scalar_type new_scale =
//                            (current_threshold > prev_threshold) ?
//                                (6.0 / (current_threshold - prev_threshold)) : 0;

//                    scale = std::max(scale, new_scale);
//                }
//                if (j + 1 < thresholds_sorting_order.size()) {
//                    const scalar_type next_threshold =
//                            feature_thresholds[thresholds_sorting_order[j + 1]];

//                    BN_VERIFY(next_threshold >= current_threshold);

//                    const scalar_type new_scale =
//                            (current_threshold < next_threshold) ?
//                                (6.0 / (next_threshold - current_threshold)) : 0;

//                    scale = std::max(scale, new_scale);
//                }

//                sigmoid_scales[j] = std::max(sigmoid_scales[j], scale);
//                BN_VERIFY(sigmoid_scales[j] > 0);
//            }
//        }

//        for (size_t j = 0; j < feature_positions.size(); ++j) {
//            const uint32_t f = feature_positions[j];

//            sigmoids.col(f) = sigmoid_scales[f] * (thresholds_[f] - features.col(features_[f]).array());

//            sigmoids.col(f) = Eigen::exp(sigmoids.col(f));
//            sigmoids.col(f) += 1;
//            sigmoids.col(f) = Eigen::inverse(sigmoids.col(f));
//        }

//        const size_t object_count = (size_t)features.rows();

//        for (size_t j = 0; j < object_count; ++j) {
//            const uint32_t this_split_index = split_indexes_(j);
//            size_t splitting_position = std::numeric_limits<size_t>::max();

//            {
//                uint32_t current_split_index = this_split_index;
//                for (size_t f : feature_positions) {
//                    current_split_index &= ~(((uint32_t)1) << (features_.size() - f - 1));
//                }

//                for (size_t k = 0; k <= thresholds_sorting_order.size(); ++k) {
//                    if (current_split_index == this_split_index) {
//                        if (k == 0) {
//                            splitting_position = feature_positions[
//                                    thresholds_sorting_order[0]];
//                        } else if (k == thresholds_sorting_order.size()) {
//                            splitting_position = feature_positions[
//                                    thresholds_sorting_order[k - 1]];
//                        } else {
//                            const auto prev_threshold = feature_thresholds[
//                                    thresholds_sorting_order[k - 1]];
//                            const auto next_threshold = feature_thresholds[
//                                    thresholds_sorting_order[k]];
//                            const auto x = features(j, features_[feature_positions.front()]);

//                            if (fabs(x - prev_threshold) <= fabs(x - next_threshold)) {
//                                splitting_position = feature_positions[
//                                        thresholds_sorting_order[k - 1]];
//                            } else {
//                                splitting_position = feature_positions[
//                                        thresholds_sorting_order[k]];
//                            }
//                        }
//                        break;
//                    }

//                    BN_VERIFY(k < thresholds_sorting_order.size());

//                    current_split_index +=
//                            (((uint32_t)1) << (features_.size() - feature_positions[
//                                               thresholds_sorting_order[k]] - 1));

//                }
//            }

//            BN_VERIFY(splitting_position < features_.size());

//            const uint32_t other_split_index =
//                    this_split_index ^
//                    (((uint32_t)1) << (features_.size() - splitting_position - 1));

//            const row_vector& this_value =
//                    is_not_initialized(shrinked_values_[this_split_index]) ?
//                        zero :
//                        shrinked_values_[this_split_index];

//            const row_vector& other_value =
//                    is_not_initialized(shrinked_values_[other_split_index]) ?
//                        zero :
//                        shrinked_values_[other_split_index];

//            if (this_split_index & (((uint32_t)1) << (features_.size() - splitting_position - 1))) {
//                gradient(j, features_[feature_positions.front()]) +=
//                        (used_out_gradient.row(j) * (
//                            this_value.array() -
//                            other_value.array())).sum() *
//                        sigmoids(j, splitting_position) * (1 - sigmoids(j, splitting_position)) *
//                        sigmoid_scales[splitting_position];
//            } else {
//                gradient(j, features_[feature_positions.front()]) +=
//                        (used_out_gradient.row(j) * (
//                            other_value.array() -
//                            this_value.array())).sum() *
//                        sigmoids(j, splitting_position) * (1 - sigmoids(j, splitting_position)) *
//                        sigmoid_scales[splitting_position];
//            }
//        }
//    }
//}

//void oblivious_tree::gradient4(const matrix& features,
//                               const matrix& out_gradient,
//                               matrix& gradient) const
//{
//    array sigmoids;
//    sigmoids.setZero(features.rows(), features_.size());

//    const array used_out_gradient = select_cols(out_gradient, used_output_features_);
//    const row_vector zero = row_vector::Zero(used_output_features_.size());

//    std::vector<bool>        processed(features_.size(), false);

//    std::vector<size_t>      feature_positions;
//    std::vector<scalar_type> feature_thresholds;
//    std::vector<size_t>      thresholds_sorting_order;

//    for (size_t i = 0; i < features_.size(); ++i) {
//        if (processed[i]) {
//            continue;
//        }

//        feature_positions.clear();
//        feature_thresholds.clear();
//        thresholds_sorting_order.clear();

//        for (size_t j = i; j < features_.size(); ++j) {
//            if (features_[j] == features_[i]) {
//                processed[j] = true;
//                feature_positions.push_back(j);
//                feature_thresholds.push_back(thresholds_[j]);
//                thresholds_sorting_order.push_back(
//                            thresholds_sorting_order.size());
//            }
//        }

//        if (thresholds_sorting_order.size() > 1) {
//            std::sort(thresholds_sorting_order.begin(),
//                      thresholds_sorting_order.end(),
//                      [&feature_thresholds](const size_t l, const size_t r) {
//                return feature_thresholds[l] < feature_thresholds[r];
//            });
//        }

//        for (uint32_t f : feature_positions) {
//            sigmoids.col(f) = (thresholds_[f] - features.col(features_[f]).array());
//            sigmoids.col(f) = Eigen::exp(sigmoids.col(f));
//            sigmoids.col(f) += 1;
//            sigmoids.col(f) = Eigen::inverse(sigmoids.col(f));
//        }

//        const size_t object_count = (size_t)features.rows();

//        for (size_t j = 0; j < object_count; ++j) {
//            const uint32_t split_index = split_indexes_(j);
//            uint32_t current_split_index = split_index;
//            for (size_t f : feature_positions) {
//                current_split_index &= ~(((uint32_t)1) << (features_.size() - f - 1));
//            }

//            for (size_t k = 0; k <= thresholds_sorting_order.size(); ++k) {
//                scalar_type sigmoid_product = 1;
//                scalar_type sigmoid_sum = 0;

//                if (!is_not_initialized(shrinked_values_[current_split_index])) {
//                    for (size_t f : feature_positions) {
//                        if (current_split_index & (((uint32_t)1) << (features_.size() - f - 1))) {
//                            sigmoid_product *= sigmoids(j, f);
//                            sigmoid_sum += (1 - sigmoids(j, f));
//                        } else {
//                            sigmoid_product *= (1 - sigmoids(j, f));
//                            sigmoid_sum -= sigmoids(j, f);
//                        }
//                    }

//                    gradient(j, features_[i]) +=
//                            (used_out_gradient.row(j) * shrinked_values_[current_split_index].array()).sum() *
//                            sigmoid_product * sigmoid_sum;
//                }

//                if (k < thresholds_sorting_order.size()) {
//                    current_split_index +=
//                            (((uint32_t)1) << (features_.size() - feature_positions[
//                                               thresholds_sorting_order[k]] - 1));
//                }
//            }
//        }
//    }
//}






//        if (config_.max_trees > 0 && trees_.size() >= config_.max_trees) {
//            trees_.front().tree.call(
//                        x_value(context),
//                        target,
//                        split_indexes);
//            trees_.pop_front();
//        }

//        if (!all_prerequisites_are_constant()) {
//            std::vector<decltype(trees_.begin())> to_delete;
//            for (auto it = trees_.begin(); it != trees_.end(); ++it) {
//                tree_holder& t = *it;

//                const scalar_type entropy = t.tree.compute_entropy(x_value(context));

//                if (entropy / t.entropy < 0.00000001) {
//                    to_delete.push_back(it);
//                }
//            }
//            for (const auto it : to_delete) {
//                if (trees_.size() <= config_.min_trees) {
//                    break;
//                }

//                it->tree.call(x_value(context), target, split_indexes);
//                trees_.erase(it);
//            }
//        }







//template<typename T>
//row_array compute_pseudo_derivative(
//        const T& output_gradient,
//        const row_array& col_means)
//{
//    if (output_gradient.cols() != col_means.cols()) {
//        throw std::logic_error("bad data");
//    }

//    row_array left_sum = row_array::Zero(output_gradient.cols());
//    row_array right_sum = row_array::Zero(output_gradient.cols());
//    size_t left_count = 0;
//    size_t right_count = 0;

//    for (Eigen::Index i = 0; i < output_gradient.rows(); ++i) {
//        for (Eigen::Index j = 0; j < output_gradient.cols(); ++j) {
//            if (output_gradient(i, j) < col_means(j)) {
//                left_sum(j) += output_gradient(i, j);
//                ++left_count;
//            } else {
//                right_sum(j) += output_gradient(i, j);
//                ++right_count;
//            }
//        }
//    }
//    if (left_count > 0) {
//        left_sum /= left_count;
//    }
//    if (right_count > 0) {
//        right_sum /= right_count;
//    }

//    return -0.25 * (right_sum - left_sum);
//}
