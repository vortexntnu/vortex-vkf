#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>

namespace Models {
template<int n_x>
using State = Eigen::Vector<double,n_x>;


template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class Model_base {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	/**
	 * @brief Parent class for modelling dynamics
	 */
	Model_base() {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param Ts Time-step
	 * @param x State
	 * @return The next state _x(k+1) = F x_k
	 */
	virtual State f(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const = 0;
	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param Ts Time-step
	 * @param x State
	 * @return The next state _x(k+1) = F x_k
	 */

	virtual Measurement h(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const = 0;
	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param Ts Time-step
	 * @param x State
	 * @return System noise covariance matrix Q
	 */
	virtual Mat_vv Q(Timestep Ts, const State& x) = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param Ts Time-step
	 * @param x State
	 * @return Measuerement noise covariance matrix R
	 */
	virtual Mat_ww R(Timestep Ts, const State& x) = 0;
};

} // namespace Models