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
	 * @param Ts Time-step
	 * @param x State
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State f(Timestep Ts, State x, Input u, Disturbance v) = 0;
	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param Ts Time-step
	 * @param x State
	 * @return The next state x_(k+1) = F x_k
	 */

	virtual Measurement h(Timestep Ts, State x, Noise w) = 0;
	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param Ts Time-step
	 * @param x State
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(Timestep Ts, State x) = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param Ts Time-step
	 * @param x State
	 * @return Measuerement noise covariance matrix R
	 */
	virtual Mat R(Timestep Ts, State x) = 0;
};

} // namespace Models