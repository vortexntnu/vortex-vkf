#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>

namespace Models {

class Model_base {
public:
	/**
	 * @brief Parent class for modelling dynamics
	 */
	Model_base() {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State f(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual Measurement h(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return Measuerement noise covariance matrix R
	 */
	virtual Mat R(std::chrono::milliseconds Ts, State x) const = 0;
};

}