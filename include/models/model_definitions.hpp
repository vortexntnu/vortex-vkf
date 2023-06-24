#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>

namespace Models {
using State       = Eigen::VectorXd;
using Measurement = Eigen::VectorXd;
using Input       = Eigen::VectorXd;
using Mat         = Eigen::MatrixXd;
using Timestep    = std::chrono::milliseconds;
} // namespace Models