#include "murtys_method_impl.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <vector>

std::vector<std::pair<double, Eigen::MatrixXd>> partition(
    const Eigen::MatrixXd& P,
    const Eigen::VectorXi& S) {
    std::vector<std::pair<double, Eigen::MatrixXd>> Q_prime;

    // Each assignment in S is excluded once to create a new subproblem
    for (int i = 0; i < S.size(); ++i) {
        // Skip invalid assignments
        if (S(i) == -1)
            continue;

        // Create a copy of the cost matrix and set a high cost for the current
        // assignment to prevent it
        Eigen::MatrixXd new_P = P;
        new_P(i, S(i)) = std::numeric_limits<double>::max();

        Q_prime.emplace_back(0, new_P);  // The cost (0) will be recalculated
                                         // when solving the subproblem
    }

    return Q_prime;
}

bool is_valid_solution(const Eigen::VectorXi& solution) {
    // A solution is valid if it does not contain any forbidden assignments
    // In this context, a forbidden assignment could be represented by -1 or any
    // other marker used
    return !solution.isConstant(-1);
}
