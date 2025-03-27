#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <memory>
#include <queue>
#include <vector>
#include <vortex_filtering/utils/algorithms/auction_algorithm.hpp>

namespace vortex::utils {

// Concept to ensure that the assignment algorithm is compatible with the cost
// matrix
template <typename T>
concept assignment_algorithm =
    requires(T a, const Eigen::MatrixXd& cost_matrix) {
        { a(cost_matrix) } -> std::same_as<std::pair<double, Eigen::VectorXi>>;
    };

std::vector<std::pair<double, Eigen::VectorXi>> murtys_method(const Eigen::MatrixXd &cost_matrix, int m, auto assignment_solver = auction_algorithm())
  requires(assignment_algorithm<decltype(assignment_solver)>)

} // namespace vortex::utils
