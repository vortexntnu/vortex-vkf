#include <limits>
#include <vector>
#include <vortex_filtering/utils/algorithms/auction_algorithm.hpp>

namespace vortex::utils {

std::pair<double, Eigen::VectorXi> auction_algorithm(const Eigen::MatrixXd &cost_matrix)
{
  int num_items              = cost_matrix.cols();
  Eigen::VectorXi assignment = Eigen::VectorXi::Constant(num_items, -1);
  Eigen::VectorXd prices     = Eigen::VectorXd::Zero(num_items);

  std::vector<int> unassigned;
  for (int i = 0; i < num_items; ++i) {
    unassigned.push_back(i);
  }

  double epsilon = 1.0 / (num_items + 1);

  while (!unassigned.empty()) {
    int person = unassigned.back();
    unassigned.pop_back();

    double max_value = std::numeric_limits<double>::lowest();
    int max_item     = -1;
    for (int item = 0; item < num_items; ++item) {
      double value = cost_matrix(person, item) - prices[item];
      if (value > max_value) {
        max_value = value;
        max_item  = item;
      }
    }

    int current_owner = assignment[max_item];
    if (current_owner != -1) {
      unassigned.push_back(current_owner);
    }

    assignment[max_item] = person;
    prices[max_item] += max_value + epsilon;
  }

  double total_cost = prices.sum();
  return {total_cost, assignment};
}

}  // namespace vortex::utils