#include "murtys_method_impl.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <vortex_filtering/utils/algorithms/murtys_method.hpp>

namespace vortex::utils {

std::vector<std::pair<double, Eigen::VectorXi>> murtys_method(const Eigen::MatrixXd &cost_matrix, int num_solutions, auto assignment_solver)
{
  std::vector<std::pair<double, Eigen::VectorXi>> R; // Solution set R
  std::set<Eigen::VectorXi> unique_solutions;        // To avoid duplicate solutions
  auto comp = [](const std::pair<double, Eigen::MatrixXd> &a, const std::pair<double, Eigen::MatrixXd> &b) { return a.first < b.first; };
  std::priority_queue<std::pair<double, Eigen::MatrixXd>, std::vector<std::pair<double, Eigen::MatrixXd>>, decltype(comp)> Q(comp);

  auto first_solution = assignment_solver(cost_matrix); // Solve initial problem
  unique_solutions.insert(first_solution.second);
  R.push_back(first_solution); // Add the first solution to R

  while (R.size() < num_solutions && !Q.empty()) {
    auto [_, P] = Q.top(); // Fetch the next problem to solve
    Q.pop();

    // Generate subproblems based on the last solution added to R
    auto subproblems = partition(P, R.back().second);

    for (auto &[__, new_P] : subproblems) {
      auto new_solution = assignment_solver(new_P); // Solve the subproblem

      // If the solution is new (unique) and valid, add it to the queue and solution set
      if (unique_solutions.find(new_solution.second) == unique_solutions.end() && is_valid_solution(new_solution.second)) {
        unique_solutions.insert(new_solution.second);
        R.push_back(new_solution);
        Q.push({new_solution.first, new_P}); // Add the new subproblem to Q with its cost
      }
    }
  }

  return R; // Return the set of solutions found
}

} // namespace vortex::utils