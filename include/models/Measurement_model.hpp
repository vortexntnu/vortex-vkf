#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
using State = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Timestep  = std::chrono::milliseconds;

namespace Model {

class Measurement_model {
public:
	/**
	 * @brief Parent class for measuerement models
	 */
	Measurement_model() {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State h(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return Measuerement noise covariance matrix R
	 */
	virtual Mat R(std::chrono::milliseconds Ts, State x) const = 0;
};

class EKF_Measurement_model : Measurement_model {
public:
	EKF_Measurement_model() : Measurement_model() {}
	virtual State h(std::chrono::milliseconds Ts, State x) const = 0;
	
	virtual Mat H(std::chrono::milliseconds Ts, State x) const = 0;

	virtual Mat R(std::chrono::milliseconds Ts, State x) const = 0;
	
};

} // End namespace Model