#pragma once
#include <eigen3/Eigen/Eigen>
#include <chrono>

namespace Models
{
using State = Eigen::VectorXd;
using Measurement = Eigen::VectorXd;
using Input = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Timestep = std::chrono::milliseconds;
}