#pragma once
#include <models/Dynamics_model.hpp>
#include <models/Measurement_model.hpp>

using State = Eigen::VectorXd;
using Measurement = Eigen::VectorXd;
using Input = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Timestep  = std::chrono::milliseconds;