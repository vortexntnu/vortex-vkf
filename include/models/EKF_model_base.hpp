#pragma once
#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>

namespace Models {

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class EKF_model_base : public Model_base<n_x,n_y,n_u,n_v,n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

	/**
	 * @brief Parent class for functions that need to be provided for the EKF filter.
	 * Children of this class will work in the EKF filter.
	 */
	EKF_model_base() : Model_base<n_x,n_y,n_u,n_v,n_w>() {}
	virtual ~EKF_model_base() {}

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat_xx F_x(Timestep Ts, State x, Input u = Input::Zero(), Disturbance v = Disturbance::Zero()) = 0;
	virtual Mat_xv F_v(Timestep Ts, State x, Input u = Input::Zero(), Disturbance v = Disturbance::Zero()) 
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return Mat_vv::Identity();
	}

	virtual Mat_yx H_x(Timestep Ts, State x, Input u = Input::Zero(), Noise w = Noise::Zero()) = 0;
	virtual Mat_yw H_w(Timestep Ts, State x, Input u = Input::Zero(), Noise w = Noise::Zero())
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)w;
		return Mat_ww::Identity();
	}
};
} // namespace Models