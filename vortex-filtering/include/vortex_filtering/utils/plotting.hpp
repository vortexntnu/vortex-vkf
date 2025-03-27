/**
 * @file utils.hpp
 * @author Eirik Kolås
 * @brief Utils for converting between types of this library and types suitable
 * for plotting with gnuplot-iostream
 * @version 0.1
 * @date 2023-11-17
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <gnuplot-iostream.h>
#include <Eigen/Dense>
#include <vector>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/utils/ellipse.hpp>

namespace vortex {
namespace plotting {

/** Convert a Gaussian to an ellipse.
 * @param gauss 2D Gaussian
 * @param scale_factor (optional) Scale factor for the ellipse
 * @return Ellipse
 */
utils::Ellipse gauss_to_ellipse(const vortex::prob::Gauss2d& gauss,
                                double scale_factor = 1.0);

/** Create a normalized-error-squared NEEDS series from a series of errors and a
 * covariance matrix.
 * @param errors
 * @param covarainces
 * @param indices (optional) Indices of the states to use for the NEEDS
 * calculation. If empty, all indices are used.
 * @return std::vector<double>
 */
template <int n_dims>
std::vector<double> create_nees_series(
    const std::vector<Eigen::Vector<double, n_dims>>& errors,
    const std::vector<Eigen::Matrix<double, n_dims, n_dims>>& covariances,
    const std::vector<size_t>& indices) {
    using Vec_n = Eigen::Vector<double, n_dims>;
    using Mat_nn = Eigen::Matrix<double, n_dims, n_dims>;

    std::vector<double> nees_series;

    for (size_t i = 0; i < errors.size(); ++i) {
        Vec_n error = errors[i];
        Mat_nn covariance = covariances[i];

        // Handle indices if provided
        if (!indices.empty()) {
            Vec_n error_sub(indices.size());
            Mat_nn covariance_sub(indices.size(), indices.size());

            for (size_t j = 0; j < indices.size(); ++j) {
                error_sub(j) = error(indices[j]);
                for (size_t k = 0; k < indices.size(); ++k) {
                    covariance_sub(j, k) = covariance(indices[j], indices[k]);
                }
            }

            error = error_sub;
            covariance = covariance_sub;
        }

        // NEEDS calculation
        double needs = error.transpose() * covariance.inverse() * error;
        nees_series.push_back(needs);
    }

    return nees_series;
}

/** Create a series of errors from a series of true states and a series of
 * estimated states.
 * @param x_true
 * @param x_est
 * @return std::vector<Eigen::Vector_n>
 */
template <int n_dims>
std::vector<Eigen::Vector<double, n_dims>> create_error_series(
    const std::vector<Eigen::Vector<double, n_dims>>& x_true,
    const std::vector<vortex::prob::Gauss<n_dims>>& x_est) {
    std::vector<Eigen::Vector<double, n_dims>> error_series;
    for (size_t i = 0; i < x_true.size(); ++i) {
        error_series.push_back(x_true[i] - x_est[i].mean());
    }
    return error_series;
}

/** Extract single state from a series of states.
 * @param x_series
 * @param index
 * @return std::vector<double>
 */
template <int n_dims>
std::vector<double> extract_state_series(
    const std::vector<Eigen::Vector<double, n_dims>>& x_series,
    size_t index) {
    std::vector<double> state_series;
    for (size_t i = 0; i < x_series.size(); ++i) {
        state_series.push_back(x_series[i](index));
    }
    return state_series;
}

/** Extract mean from a series of Gaussians.
 * @param x_series
 * @return std::vector<Eigen::Vector_n>
 */
template <int n_dims>
std::vector<Eigen::Vector<double, n_dims>> extract_mean_series(
    const std::vector<vortex::prob::Gauss<n_dims>>& x_series) {
    std::vector<Eigen::Vector<double, n_dims>> mean_series;
    for (size_t i = 0; i < x_series.size(); ++i) {
        mean_series.push_back(x_series[i].mean());
    }
    return mean_series;
}

/** Approximate Gaussian from samples.
 * @param samples
 * @return vortex::prob::Gauss_n
 */
template <int n_dims>
vortex::prob::Gauss<n_dims> approximate_gaussian(
    const std::vector<Eigen::Vector<double, n_dims>>& samples) {
    using Vec_n = Eigen::Vector<double, n_dims>;
    using Mat_nn = Eigen::Matrix<double, n_dims, n_dims>;

    Vec_n mean = Vec_n::Zero();
    for (const auto& sample : samples) {
        mean += sample;
    }
    mean /= samples.size();

    Mat_nn cov = Mat_nn::Zero();
    for (const auto& sample : samples) {
        cov += (sample - mean) * (sample - mean).transpose();
    }
    cov /= samples.size();

    return {mean, cov};
}

}  // namespace plotting
}  // namespace vortex
