#pragma once
#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>

namespace Models {

template<int n_x, int n_y, int n_u, int n_v, int n_w>
class EKF_model_base : public Model_base<n_x,n_y,n_u,n_v,n_w> {
public:
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::State;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Measurement;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Input;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Disturbance;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Noise;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_xx; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yy; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_vv; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_ww; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_xv;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_xu;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yx;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yu;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yw;

	/**
	 * @brief Parent class for functions that need to be provided for the EKF filter.
	 */
	EKF_model_base() : Model_base<n_x,n_y,n_u,n_v,n_w>{} {}

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat_xx F_x(Timestep Ts, State x, Input u, Disturbance v) = 0;
	virtual Mat_xv F_v(Timestep Ts, State x, Input u, Disturbance v) 
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return Mat_xv::Identity();
	}

	virtual Mat_yx H_x(Timestep Ts, State x, Input u, Noise w) = 0;
	virtual Mat_yw H_w(Timestep Ts, State x, Input u, Noise w)
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)w;
		return Mat_yw::Identity();
	}
};
} // namespace Models