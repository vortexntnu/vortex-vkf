#pragma once
#include <models/model_definitions.hpp>
#include <models/Model_base.hpp>

namespace Models
{
class EKF_model_base : public Model_base {
public:
	/**
	 * @brief Parent class for functions that need to be provided for the EKF filter.
	 */
	EKF_model_base() : Model_base{} {}
	
	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State f(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State h(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat F(std::chrono::milliseconds Ts, State x) const = 0;

    virtual Mat H(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(std::chrono::milliseconds Ts, State x) const = 0;

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