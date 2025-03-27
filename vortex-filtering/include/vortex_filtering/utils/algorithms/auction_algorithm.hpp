#pragma once
#include <Eigen/Dense>
#include <vector>

namespace vortex::utils {

std::pair<double, Eigen::VectorXi> auction_algorithm(
    const Eigen::MatrixXd& cost_matrix);

}  // namespace vortex::utils
