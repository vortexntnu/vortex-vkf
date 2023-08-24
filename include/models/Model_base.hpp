#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>
#include <integration_methods/ERK_methods.hpp>

#include <memory>
using std::placeholders::_1;
using std::placeholders::_2;
namespace Models {

template <class integrator, int n_x, int n_y, int n_u, int n_v = n_x, int n_w = n_y> 
class Model_base {
public:
	// Make static constexpr constants so that the model itself may be used to infer the sizes instead of individual parameters
	static constexpr int _n_x = n_x, _n_y = n_y, _n_u = n_u, _n_v = n_v, _n_w = n_w;

	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)
	/**
	 * @brief Parent class for modelling dynamics
	 */
	Model_base(Mat_vv Q = Mat_vv::Identity(), Mat_ww R = Mat_ww::Identity()) : _Q{Q}, _R{R} {}

	/**
	 * @brief Prediction equation f(t,x,u,v) = x_dot. (Use Intergator::None for discrete time)
	 * @param t Simulation time
	 * @param x State
	 * @param u Input
	 * @param v Disturbance
	 * @return The derivative x_dot = f(t,x,u,v)
	 */
	virtual State f(Time t, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const = 0;

	/**
	 * @brief Integrate the model from time \p t0 to \p t0 + \p dt
	 * @param dt Timestep
	 * @param t0 Start time
	 * @param x0 Initial state
	 * @param u Input
	 * @param v Disturbance
	 * @return The next state x_(k+1) = F(x_k,...)
	 */
	virtual State next_state(Timestep dt, Time t0, const State &x0, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const
	{
		// Cast f to a lambda function to keep u and v constant
		State_dot F = [this, &u, &v](Time t, const State &x) { return this->f(t, x, u, v); };
		return integrator::integrate(F, dt, t0, x0);
	}
	/**
	 * @brief Measurement equation h:
	 * Calculate the zero noise prediction at time \p t from \p x.
	 * @param t Simulation time
	 * @param x State
	 * @return The prediction y = h(t,x,u,w)
	 */
	virtual Measurement h(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const = 0;

	/**
	 * @brief Covariance matrix of model disturbance:
	 * Calculate the transition covariance \p Q for time \p t
	 * @param t Simulation time
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
	 * @param t Simulation time
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