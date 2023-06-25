#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>

namespace Models {
using State       = Eigen::VectorXd;
using Measurement = Eigen::VectorXd;
using Input       = Eigen::VectorXd;
using Disturbance = Eigen::VectorXd;
using Noise       = Eigen::VectorXd;
using Mat         = Eigen::MatrixXd;
using Timestep    = std::chrono::milliseconds;
using namespace std::chrono_literals;
} // namespace Models