#include <Eigen/Dense>
#include <set>
#include <vector>

std::vector<std::pair<double, Eigen::MatrixXd>> partition(
    const Eigen::MatrixXd& P,
    const Eigen::VectorXi& S);

bool is_valid_solution(const Eigen::VectorXi& solution);
