#pragma once
#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>

namespace Models {

template <class Integrator, int n_x, int n_y, int n_u, int n_v = n_x, int n_w = n_y> 
class EKF_model_base : public Model_base<Integrator, n_x, n_y, n_u, n_v, n_w> {
public:
	// These type definitions are needed because of the stupid two-phase lookup for dependent names in templates in C++
	using Base = Model_base<Integrator, n_x, n_y, n_u, n_v, n_w>;
	using Base::_n_x; using Base::_n_y; using Base::_n_u; using Base::_n_v; using Base::_n_w;
	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)

	/**
	 * @brief Parent class for functions that need to be provided for the EKF filter.
	 * All children of this class will work in the EKF as well as in the UKK. 
	 * For the latter case it is recommended to explicitly specify disturbance and noise as additive. (i.e. f(x)+v and h(x)+w)
	 */
	EKF_model_base() : Base() {}
	virtual ~EKF_model_base() {}

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p t at \p x
	 * @param x State
	 * @param t Time-step
	 * @return Jacobian A
	 */
	virtual Mat_xx F_x(Time t, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const = 0;
	/**
	 * @brief Jacobian of f with respect to v:
	 * Calculate the transition function jacobian for time \p t at \p x
	 * This has to be defined if the disturbance on the model is not additive.
	 * @param t Time
	 * @param x State
	 * @param u Input
	 * @param v Disturbance
	 * @return Jacobian G
	 */
	virtual Mat_xv F_v(Time t, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const
	{
		(void)t;
		(void)x;
		(void)u;
		(void)v;
		return Mat_xv::Identity();
	}
	/**
	 * @brief Jacobian of h:
	 * Calculate the measurement function jacobian for time \p t at \p x
	 * @param t Time-step
	 * @param x State
	 * @return Jacobian C
	 */
	virtual Mat_yx H_x(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const = 0;

	/**
	 * @brief Jacobian of h with respect to w:
	 * Calculate the measurement function jacobian for time \p t at \p x
	 * This has to be defined if the noise on the measurement is not additive.
	 * @param t Time
	 * @param x State
	 * @param u Input
	 * @param w Noise
	 * @return Jacobian H
	 */
	virtual Mat_yw H_w(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const
	{
		(void)t;
		(void)x;
		(void)u;
		(void)w;
		return Mat_yw::Identity();
	}
};



template <class Integrator, int n_x, int n_y, int n_u, int n_v = n_x, int n_w = n_y> 
class LTI_model : public EKF_model_base<Integrator, n_x, n_y, n_u, n_v, n_w> {
public:
	// These type definitions are needed because of the stupid two-phase lookup for dependent names in templates in C++
	using Base = EKF_model_base<Integrator, n_x, n_y, n_u, n_v, n_w>;
	using Base::_n_x; using Base::_n_y; using Base::_n_u; using Base::_n_v; using Base::_n_w;
	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)

	/**
	 * @brief Construct a lti model object
	 * 
	 * @param A System matrix
	 * @param B Input matrix
	 * @param C Measurement matrix
	 * @param D Feedthrough matrix
	 * @param Q Disturbance covariance matrix
	 * @param R Measurement covariance matrix
	 * @param G Disturbance input matrix
	 * @param H Noise input matrix
	 */
	LTI_model(Mat_xx A, Mat_xu B, Mat_yx C, Mat_yu D, Mat_vv Q, Mat_ww R, Mat_xv G, Mat_yw H)
	    : Base(), _A{A}, _B{B}, _C{C}, _D{D}, _Q{Q}, _R{R}, _G{G}, _H{H}
	{
	}
	/**
	 * @brief Construct a lti model object with no feedthrough and no noise input
	 * 
	 * @param A System matrix
	 * @param B Input matrix
	 * @param C Measurement matrix
	 * @param Q Disturbance covariance matrix
	 * @param R Measurement covariance matrix
	 */
	LTI_model(Mat_xx A, Mat_xu B, Mat_yx C, Mat_vv Q, Mat_ww R) : LTI_model(A, B, C, Mat_yu::Zero(), Q, R, Mat_xv::Identity(), Mat_yw::Identity()) {}

	~LTI_model() {}

	/**
	 * @brief Prediction equation f(t,x,u,v) = x_dot. (Use intergator::None for discrete time)
	 * @param t Simulation time
	 * @param x State
	 * @param u Input
	 * @param v Disturbance
	 * @return The derivative x_dot = f(t,x,u,v)
	 */
	State f(Time t, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const override final
	{
		(void)t;
		return _A * x + _B * u + _G * v;
	}
	/**
	 * @brief Measurement equation h:
	 * Calculate the zero noise prediction at time \p t from \p x.
	 * @param t Simulation time
	 * @param x State
	 * @return The prediction y = h(t,x,u,w)
	 */
	Measurement h(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const override final
	{
		(void)t;
		return _C * x + _D * u + _H * w;
	}
	/**
	 * @brief Returns A: the jacobian of f with respect to x
	 * 
	 * @param t Time
	 * @param x State
	 * @param u Input
	 * @param v Disturbance
	 * @return A
	 */
	Mat_xx F_x(Time t, const State &x, const Input &u, const Disturbance &v = Disturbance::Zero()) const override final
	{
		(void)t;
		(void)x;
		(void)u;
		(void)v;
		return _A;
	}

	/**
	 * @brief Returns G: the jacobian of f with respect to v
	 * @param t Time
	 * @param x State
	 * @param u Input
	 * @param v Disturbance
	 * @return G
	 */
	Mat_xv F_v(Time t, const State &x, const Input &u, const Disturbance &v = Disturbance::Zero()) const override final
	{
		(void)t;
		(void)x;
		(void)u;
		(void)v;
		return _G;
	}
	/**
	 * @brief Returns C: the jacobian of h with respect to x
	 * 
	 * @param t Time
	 * @param x State
	 * @param u Input
	 * @param w Noise
	 * @return Mat_yx 
	 */
	Mat_yx H_x(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const override final
	{
		(void)t;
		(void)x;
		(void)u;
		(void)w;
		return _C;
	}
	/**
	 * @brief Returns H: the jacobian of h with respect to w
	 * 
	 * @param t Time
	 * @param x State
	 * @param u Input
	 * @param w Noise
	 * @return Mat_yw 
	 */
	Mat_yw H_w(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const override final
	{
		(void)x;
		(void)u;
		(void)t;
		(void)w;
		return _H;
	}
	/**
	 * @brief Covariance matrix of model disturbance:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Simulation time
	 * @param x State
	 * @return System noise covariance matrix Q
	 */
	const Mat_vv &Q(Time t, const State &x) const override final
	{
		(void)t;
		(void)x;
		return _Q;
	}
	/**
	 * @brief Covariance matrix of measurement noise:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Simulation time
	 * @param x State
	 * @return Measuerement noise covariance matrix R
	 */
	const Mat_ww &R(Time t, const State &x) const override final
	{
		(void)t;
		(void)x;
		return _R;
	}

	Mat_xx _A;
	Mat_xu _B;
	Mat_yx _C;
	Mat_yu _D;
	Mat_vv _Q;
	Mat_yy _R;
	Mat_xv _G;
	Mat_yw _H;
};


} // namespace Models