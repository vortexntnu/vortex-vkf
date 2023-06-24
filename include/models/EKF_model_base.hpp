#pragma once
#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>

namespace Models {
class EKF_model_base : public Model_base {
public:
	/**
	 * @brief Parent class for functions that need to be provided for the EKF filter.
	 */
	EKF_model_base() : Model_base{} {}

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat F(Timestep Ts, State x) = 0;

	virtual Mat H(Timestep Ts, State x) = 0;
};
} // namespace Models