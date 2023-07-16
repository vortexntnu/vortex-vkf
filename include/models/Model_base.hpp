#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>

namespace Models {


template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class Model_base {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	/**
	 * @brief Parent class for modelling dynamics
	 */
	Model_base(Mat_vv Q = Mat_vv::Identity(), Mat_ww R = Mat_ww::Identity()) :  _Q{Q}, _R{R} {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p t from \p x.
	 * @param t Time-step
	 * @param x State
	 * @return The next state _x(k+1) = F x_k
	 */
	virtual State f(Time t, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const = 0;

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p t from \p x.
	 * @param t Time-step
	 * @param x State
	 * @return The next state _x(k+1) = F x_k
	 */
	virtual Measurement h(Time t, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Time-step
	 * @param x State
	 * @return System noise covariance matrix Q
	 */
	virtual const Mat_vv& Q(Time t, const State& x) const
	{
		(void)t;
		(void)x;
		return _Q;
	}

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Time-step
	 * @param x State
	 * @return Measuerement noise covariance matrix R
	 */
	virtual const Mat_ww& R(Time t, const State& x) const
	{
		(void)t;
		(void)x;
		return _R;
	}

	Mat_vv _Q;
	Mat_ww _R;
};

} // namespace Models