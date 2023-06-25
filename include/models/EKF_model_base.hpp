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
	virtual Mat F_x(Timestep Ts, State x, Input u, Disturbance v) = 0;
	virtual Mat F_v(Timestep Ts, State x, Input u, Disturbance v) 
	{
		(void)Ts;
		(void)u;
		(void)v;
		Mat I = Eigen::MatrixXd::Identity(x.rows(),x.rows());
		return I;
	}

	virtual Mat H_x(Timestep Ts, State x, Noise w) = 0;
	virtual Mat H_w(Timestep Ts, State x, Noise w)
	{
		(void)Ts;
		(void)w;
		Mat I = Eigen::MatrixXd::Identity(x.rows(),x.rows());
		return I;
	}
};
} // namespace Models