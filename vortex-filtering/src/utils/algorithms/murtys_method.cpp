#include "murtys_method_impl.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <vortex_filtering/utils/algorithms/murtys_method.hpp>

namespace vortex::utils {

using CostValuePair = std::pair<double, Eigen::VectorXi>;

std::vector<CostValuePair> murtys_method(const Eigen::MatrixXd& cost_matrix,
                                         int num_solutions,
                                         auto assignment_solver) {
    std::vector<CostValuePair> R;  // Solution set R
    auto comp = [](const CostValuePair& a, const CostValuePair& b) {
        return a.first < b.first;
    };
    std::priority_queue<CostValuePair, std::vector<CostValuePair>,
                        decltype(comp)>
        Q(comp);

    Q.push(assignment_solver(
        cost_matrix));  // Add the first problem to Q with its cost

    while (R.size() < num_solutions && !Q.empty()) {
        auto [_, Q_max] = Q.top();  // Fetch the next problem to solve
        Q.pop();
        R.push_back(Q_max);
        // Generate subproblems based on the last solution added to R
        auto Q_ = partition(Q_max.first, Q_max.second);
    }

    return R;  // Return the set of solutions found
}

}  // namespace vortex::utils
