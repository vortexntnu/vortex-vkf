/**
 * @file utils.hpp
 * @author Eirik Kolås
 * @brief Utils for converting between types of this library and types suitable for plotting with gnuplot-iostream
 * @version 0.1
 * @date 2023-11-17
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <Eigen/Dense>
#include <gnuplot-iostream.h>
#include <vector>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace plotting {

struct Ellipse {
	double x;     // center
	double y;     // center
	double a;     // major axis length
	double b;     // minor axis length
	double angle; // angle in degrees
};

/** Convert a Gaussian to an ellipse.
 * @param MultiVarGauss
 * @return Ellipse
 */
Ellipse gauss_to_ellipse(const vortex::prob::Gauss2d &gauss);

/** Create a normalized-error-squared NEES series from a series of errors and a covariance matrix.
 * @param errors
 * @param covarainces
 * @param indices (optional) Indices of the states to use for the NEES calculation. If empty, all indices are used.
 * @return std::vector<double>
 */
std::vector<double> create_nees_series(const std::vector<Eigen::VectorXd> &errors, const std::vector<Eigen::MatrixXd> &covarainces,
                                       const std::vector<size_t> &indices = std::vector<size_t>());

/** Create a series of errors from a series of true states and a series of estimated states.
 * @param x_true
 * @param x_est
 * @return std::vector<Eigen::VectorXd>
 */
std::vector<Eigen::VectorXd> create_error_series(const std::vector<Eigen::VectorXd> &x_true, const std::vector<vortex::prob::GaussXd> &x_est);

/** Extract single state from a series of states.
 * @param x_series
 * @param index
 * @return std::vector<double>
 */
std::vector<double> extract_state_series(const std::vector<Eigen::VectorXd> &x_series, size_t index);

/** Extract mean from a series of Gaussians.
 * @param x_series
 * @return std::vector<Eigen::VectorXd>
 */
std::vector<Eigen::VectorXd> extract_mean_series(const std::vector<vortex::prob::GaussXd> &x_series);

/** Approximate Gaussian from samples.
 * @param samples
 * @return vortex::prob::GaussXd
 */
vortex::prob::GaussXd approximate_gaussian(const std::vector<Eigen::VectorXd> &samples);

} // namespace plotting
} // namespace vortex
