#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>

namespace Models {

template <int n_x, int n_y, int n_u, int n_v = n_x, int n_w = n_y> class Model_base {
public:
	static constexpr int _Nx = n_x, _Ny = n_y, _Nu = n_u, _Nv = n_v, _Nw = n_w;

	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)
	/**
	 * @brief Parent class for modelling dynamics
	 */
	Model_base(Mat_vv Q = Mat_vv::Identity(), Mat_ww R = Mat_ww::Identity()) : _Q{Q}, _R{R} {}

	/**
	 * @brief Prediction equation f: Can be either continuous or discrete time, depending on the integrator that is used.
	 * Calculate the zero noise prediction at time \p t from \p x.
	 * @param t Time-step
	 * @param x State
	 * @return The next state _x(k+1) = F x_k
	 */
	virtual State f(Time t, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const = 0;

	/**
	 * @brief Measurement equation h:
	 * Calculate the zero noise prediction at time \p t from \p x.
	 * @param t Time-step
	 * @param x State
	 * @return The next state _x(k+1) = F x_k
	 */
	virtual Measurement h(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const = 0;

	/**
	 * @brief Covariance matrix of model disturbance:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Time-step
	 * @param x State
	 * @return System noise covariance matrix Q
	 */
	virtual const Mat_vv &Q(Time t, const State &x) const
	{
		(void)t;
		(void)x;
		return _Q;
	}

	/**
	 * @brief Covariance matrix of measurement noise:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Time-step
	 * @param x State
	 * @return Measuerement noise covariance matrix R
	 */
	virtual const Mat_ww &R(Time t, const State &x) const
	{
		(void)t;
		(void)x;
		return _R;
	}

	Mat_vv _Q;
	Mat_ww _R;
};

} // namespace Models