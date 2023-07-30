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
	 * Calculate the transition function jacobian for time \p t at \p x
	 * @param x State
	 * @param t Time-step
	 * @return Jacobian F
	 */
	virtual Mat_xx F_x(Time t, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const = 0;
	virtual Mat_xv F_v(Time t, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const 
	{
		(void)t;
		(void)x;
		(void)u;
		(void)v;
		return Mat_xv::Identity();
	}

	virtual Mat_yx H_x(Time t, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const = 0;
	virtual Mat_yw H_w(Time t, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const 
	{
		(void)t;
		(void)x;
		(void)u;
		(void)w;
		return Mat_yw::Identity();
	}
};
} // namespace Models