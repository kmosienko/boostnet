#pragma once

#include "model.h"

#include <vector>
#include <utility>
#include <string>

namespace boostnet {

std::vector<std::vector<size_t>> generate_minibatches(size_t from, size_t by);

matrix select_rows(const matrix& m, const std::vector<size_t>& rows);

size_t get_input_size(const std::vector<std::pair<const function_ptr&, const matrix&>>& model_train_input);

std::pair<float, float> compute_mean_stddev(const std::vector<float>& data);

struct optimizing_history {
    std::vector<float>                   test_losses;
    std::vector<std::pair<float, float>> train_losses;
};

optimizing_history sgd(
        model& model_to_optimize,
        const std::vector<std::pair<const function_ptr&, const matrix&>>& model_train_input,
        const std::vector<std::pair<const function_ptr&, const matrix&>>& model_test_input,
        const size_t minibatch_size,
        const size_t epochs,
        const float learning_rate,
        const std::string& log_file_name = "");

}
