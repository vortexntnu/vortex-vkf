#pragma once
#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>

namespace Models {

template<int n_x, int n_y, int n_u, int n_v>
class EKF_model_base : public Model_base<n_x, n_y, n_u, n_v> {
public:
	using typename Model_base<n_x,n_y,n_u,n_v>::State;
	using typename Model_base<n_x,n_y,n_u,n_v>::Measurement;
	using typename Model_base<n_x,n_y,n_u,n_v>::Input;
	using typename Model_base<n_x,n_y,n_u,n_v>::Disturbance;
	using typename Model_base<n_x,n_y,n_u,n_v>::Noise;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_vv; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_yy; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_vv; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_ww; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_xv;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_xu;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_yx;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_yw;

	/**
	 * @brief Parent class for functions that need to be provided for the EKF filter.
	 */
	EKF_model_base() : Model_base<n_x, n_y, n_u, n_v>{} {}

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat_vv F_x(Timestep Ts, State x, Input u, Disturbance v) = 0;
	virtual Mat_xv F_v(Timestep Ts, State x, Input u, Disturbance v) 
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return Mat_vv::Identity();
	}

	virtual Mat_yx H_x(Timestep Ts, State x, Noise w) = 0;
	virtual Mat_yw H_w(Timestep Ts, State x, Noise w)
	{
		(void)Ts;
		(void)x;
		(void)w;
		return Mat_yy::Identity();
	}
};
} // namespace Models