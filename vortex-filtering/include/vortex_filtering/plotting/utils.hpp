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

#include <vector>
#include <gnuplot-iostream.h>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace plotting {

struct Ellipse {
    double x; // center
    double y; // center
    double a; // major axis length
    double b; // minor axis length
    double angle; // angle in degrees
};

/** Convert a Gaussian to an ellipse.
 * @param MultiVarGauss 
 * @return Ellipse 
 */
Ellipse gauss_to_ellipse(const vortex::prob::MultiVarGauss<2>& gauss);

/** Create a normalized-error-squared NEES series from a series of errors and a covariance matrix.
 * @param errors 
 * @param covarainces
 * @param indices (optional) Indices of the states to use for the NEES calculation. If empty, all indices are used.
 * @return std::vector<double> 
 */
std::vector<double> create_nees_series(const std::vector<Eigen::VectorXd>& errors, const std::vector<Eigen::MatrixXd>& covarainces, const std::vector<size_t>& indices = std::vector<size_t>());













}  // namespace plotting
}  // namespace vortex
