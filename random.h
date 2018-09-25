#pragma once

#include "util.h"

#include <eigen3/Eigen/Eigen>

#include <random>

namespace boostnet {

template<typename T>
void fill_normal(T& array, const double mean = 0, const double stddev = 1) {
    static std::default_random_engine random_engine;
    std::normal_distribution<double> distribution(mean, stddev);

    for (int i = 0; i < array.rows(); ++i) {
        for (int j = 0; j < array.cols(); ++j) {
            array(i, j) = distribution(random_engine);
        }
    }
}

template<typename T>
T fill_normal(
        const size_t rows, const size_t cols,
        const double mean = 0, const double stddev = 1)
{
    T array(rows, cols);
    fill_normal(array, mean, stddev);
    return array;
}

inline scalar_type normal_scalar(const double mean = 0, const double stddev = 1) {
    column_array n(1);
    fill_normal(n, mean, stddev);
    return n(0);
}

inline scalar_type uniform_scalar() {
    return (rand() % 1009) / 1008.0;
}

}
