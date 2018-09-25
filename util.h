#pragma once

#include <eigen3/Eigen/Eigen>

#include <chrono>
#include <exception>
#include <stdexcept>
#include <limits>
#include <iostream>

namespace boostnet {

template<typename T>
struct reversed {
    reversed(const T& container) : container_(container) {}

    typename T::const_reverse_iterator begin() const { return container_.rbegin(); }
    typename T::const_reverse_iterator end() const { return container_.rend(); }

    const T& container_;
};


template<typename T>
using matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template<typename T>
using array_t  = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
using column_vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T>
using row_vector_t    = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template<typename T>
using column_array_t = Eigen::Array<T, Eigen::Dynamic, 1>;
template<typename T>
using row_array_t    = Eigen::Array<T, 1, Eigen::Dynamic>;


using scalar_type = double;


using matrix = matrix_t<scalar_type>;
using array  = array_t<scalar_type>;

using column_vector = column_vector_t<scalar_type>;
using row_vector    = row_vector_t<scalar_type>;

using column_array = column_array_t<scalar_type>;
using row_array    = row_array_t<scalar_type>;


template<typename TimeT = std::chrono::microseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F func, Args&&... args) {
        auto start = std::chrono::system_clock::now();

        func(std::forward<Args>(args)...);

        auto duration = std::chrono::duration_cast<TimeT>
                            (std::chrono::system_clock::now() - start);

        return duration.count();
    }

    static typename TimeT::rep current_time() {
        auto start = std::chrono::system_clock::from_time_t(0);

        const auto time = std::chrono::duration_cast<TimeT>(
                    std::chrono::system_clock::now() - start);

        return time.count();
    }
};

inline void check_or_throw(
        const bool condition,
        const char* const message)
{
    if (!condition) {
        throw std::logic_error(message);
    }
}

template<typename T>
bool is_not_initialized(const T& array) {
    return array.rows() == 0 || array.cols() == 0;
}

template<typename T>
bool is_zero(const T& m) {
    if ((m.array().abs() <= std::numeric_limits<scalar_type>::epsilon()).all()) {
        return true;
    } else {
        return false;
    }
}

} // boostnet

#define BN_VERIFY(C) boostnet::check_or_throw(C, #C)
#define BN_VERIFYM(C, M) boostnet::check_or_throw(C, M)
