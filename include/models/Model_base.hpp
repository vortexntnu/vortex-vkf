#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>

namespace Models {
template<int n_x>
using State = Eigen::Vector<double,n_x>;


template<int n_x, int n_y, int n_u, int n_v, int n_w>
class Model_base {
public:
	using State 	  = Eigen::Vector<double,n_x>;
	using Measurement = Eigen::Vector<double,n_y>;
	using Input       = Eigen::Vector<double,n_u>;
	using Disturbance = Eigen::Vector<double,n_v>;
	using Noise       = Eigen::Vector<double,n_y>; 
	using Mat_xx      = Eigen::Matrix<double,n_x,n_x>; 
	using Mat_yy      = Eigen::Matrix<double,n_y,n_y>; 
	using Mat_vv      = Eigen::Matrix<double,n_v,n_v>; 
	using Mat_ww      = Eigen::Matrix<double,n_w,n_w>; 
	using Mat_xv 	  = Eigen::Matrix<double,n_x,n_v>;
	using Mat_xu 	  = Eigen::Matrix<double,n_x,n_u>;
	using Mat_yx 	  = Eigen::Matrix<double,n_y,n_x>;
	using Mat_yu 	  = Eigen::Matrix<double,n_y,n_u>;
	using Mat_yw 	  = Eigen::Matrix<double,n_y,n_w>;

	/**
	 * @brief Parent class for modelling dynamics
	 */
	Model_base() {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param Ts Time-step
	 * @param x State
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State f(Timestep Ts, State x, Input u, Disturbance v) = 0;
	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param Ts Time-step
	 * @param x State
	 * @return The next state x_(k+1) = F x_k
	 */

	virtual Measurement h(Timestep Ts, State x, Input u, Noise w) = 0;
	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param Ts Time-step
	 * @param x State
	 * @return System noise covariance matrix Q
	 */
	virtual Mat_vv Q(Timestep Ts, State x) = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param Ts Time-step
	 * @param x State
	 * @return Measuerement noise covariance matrix R
	 */
	virtual Mat_ww R(Timestep Ts, State x) = 0;
};

} // namespace Models