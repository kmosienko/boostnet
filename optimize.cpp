#include "optimize.h"
#include "util.h"

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace boostnet {

std::vector<std::vector<size_t>> generate_minibatches(size_t from, size_t by) {
    if (from < by) {
        throw std::logic_error("from < by");
    }
    std::vector<size_t> indexes;
    for (size_t i = 0; i < from; ++i) {
        indexes.push_back(i);
    }
    for (size_t i = 0; i < from && indexes.size() % by != 0; ++i) {
        indexes.push_back(i);
    }
    if (indexes.size() % by != 0) {
        throw std::logic_error("bad algorithm");
    }

    std::random_shuffle(indexes.begin(), indexes.end());

    std::vector<std::vector<size_t>> res(indexes.size() / by, std::vector<size_t>(by, 0));
    size_t index = 0;

    for (size_t i = 0; i < res.size(); ++i) {
        for (size_t j = 0; j < by; ++j) {
            res.at(i).at(j) = indexes.at(index++);
        }
    }

    if (res.empty() || res.size() < from / by) {
        throw std::logic_error("bad algorithm");
    }
    for (const auto& batch : res) {
        if (batch.size() != by) {
            throw std::logic_error("bad algorithm");
        }
    }

    return res;
}

matrix select_rows(const matrix& m, const std::vector<size_t>& rows) {
    matrix res(rows.size(), m.cols());

    for (size_t i = 0; i < rows.size(); ++i) {
        res.row(i) = m.row(rows.at(i));
    }

    return res;
}

size_t get_input_size(const std::vector<std::pair<const function_ptr&, const matrix&>>& model_train_input) {
    if (model_train_input.empty()) {
        return 0;
    }
    int64_t res = -1;
    for (const std::pair<function_ptr, const matrix&>& p : model_train_input) {
        const matrix& m = p.second;
        if (res == -1) {
            res = m.rows();
        } else if (res != m.rows()) {
            throw std::logic_error("input arrays of different sizes");
        }
    }
    return res;
}

std::pair<float, float> compute_mean_stddev(const std::vector<float>& data) {
    if (data.empty()) {
        return {0.0f, 0.0f};
    }
    if (data.size() == 1) {
        return {data[0], 0.0f};
    }

    double mean = 0;
    double sq_mean = 0;

    for (const auto d : data) {
        mean += d;
        sq_mean += (d * d);
    }

    mean /= data.size();
    sq_mean /= data.size();

    return {mean, sqrt(std::max(0.0, sq_mean - mean * mean))};
}

template<typename T>
int to_procent(const T n, const T total) {
    if (n == total) {
        return 100;
    }
    const double p = (n / (double)total);
    return p * 100;
}

optimizing_history sgd(
        model& model_to_optimize,
        const std::vector<std::pair<const function_ptr&, const matrix&>>& model_train_input,
        const std::vector<std::pair<const function_ptr&, const matrix&>>& model_test_input,
        const size_t minibatch_size,
        const size_t epochs,
        const float learning_rate,
        const std::string& log_file_name)
{
    const auto start_time = measure<>::current_time();

    std::ofstream log;
    if (!log_file_name.empty()) {
        log.open(log_file_name);
    }

    const size_t input_size = get_input_size(model_train_input);

    optimizing_history history;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        const std::vector<std::vector<size_t>> minibatches =
                generate_minibatches(input_size, minibatch_size);

        std::cerr << "mini-batches: ";

        std::vector<float> losses;
        size_t minibatch_index = 0;
        int prev_procent_done = -1;
        for (const std::vector<size_t>& indexes : minibatches) {
            for (const std::pair<const function_ptr&, const matrix&>& p : model_train_input) {
                model_to_optimize.move_input(
                            p.first,
                            select_rows(p.second, indexes));
            }

            const auto step = [&model_to_optimize, &learning_rate]() {
                model_to_optimize.forward();
                model_to_optimize.backward();
                model_to_optimize.tune(learning_rate);
            };

            const auto step_duration = measure<>::execution(step);

            losses.push_back(model_to_optimize.get_plain_output());

            const int procent_done = to_procent(++minibatch_index, minibatches.size());

            if (prev_procent_done != procent_done) {
                std::cerr << "\33[2K\r" << (step_duration / 1000) << " ms per minibatch\t" <<
                             "target " << model_to_optimize.get_plain_output() << '\t' <<
                             procent_done << '%';

                prev_procent_done = procent_done;
            }

            if (log.is_open()) {
                const auto total_time = measure<>::current_time() - start_time;

                log << "minibatch" << '\t' <<
                       (total_time / 1000000.0) << '\t' <<
                       model_to_optimize.get_plain_output() << '\t' <<
                       0 << '\t' <<
                       0 <<
                       std::endl;
            }
        }

        model_to_optimize.set_input(model_test_input);
        model_to_optimize.forward();

        history.test_losses.push_back(model_to_optimize.get_plain_output());
        history.train_losses.push_back(compute_mean_stddev(losses));

        const auto total_time = measure<>::current_time() - start_time;

        std::cerr << "\33[2K\r" <<
                     "epoch " << epoch << ":" <<
                     " train " << history.train_losses.back().first << " +- " << 3 * history.train_losses.back().second <<
                     " test " << history.test_losses.back() <<
                     " total " << (total_time / 1000000.0) << " s" <<
                     std::endl;

        if (log.is_open()) {
            const auto total_time = measure<>::current_time() - start_time;

            log << "train" << '\t' <<
                   (total_time / 1000000.0) << '\t' <<
                   history.train_losses.back().first << '\t' <<
                   std::max(0.0f, history.train_losses.back().first - 3 * history.train_losses.back().second) << '\t' <<
                   history.train_losses.back().first + 3 * history.train_losses.back().second <<
                   std::endl;

            log << "test" << '\t' <<
                   (total_time / 1000000.0) << '\t' <<
                   history.test_losses.back() << '\t' <<
                   0 << '\t' <<
                   0 <<
                   std::endl;
        }
    }

    return history;
}

}
