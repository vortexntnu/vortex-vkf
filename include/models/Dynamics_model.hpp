#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
using State = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

namespace Model {

class Dynamics_model {
public:
	/**
	 * @brief Parent class for modelling dynamics
	 */
	Dynamics_model() {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State f(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(std::chrono::milliseconds Ts, State x) const = 0;
};

class EKF_Dynamics_model : public Dynamics_model {
public:
	/**
	 * @brief Parent class for Linear time varying models
	 * Instead of defining the system function \p f, define the Jacobian \p F
	 */
	EKF_Dynamics_model() : Dynamics_model{} {}
	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual State f(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat F(std::chrono::milliseconds Ts, State x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(std::chrono::milliseconds Ts, State x) const = 0;
};

constexpr size_t LM_size{7}; // Size of the landmark model (x, y, z and quaternion)
class Landmark : public EKF_Dynamics_model {
public:
	/**
	 * @brief Model for stationary objects with x, y, z and quaternion as states
	 * @param Q_weights The weights of the system noise covariance matrix
	 */
	Landmark(State Q_weights) : EKF_Dynamics_model{}, Q_matrix{Q_weights.asDiagonal()} {}

	State f(std::chrono::milliseconds Ts, State x) const override
	{
		(void)Ts;
		return x;
	}


	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	Mat F(std::chrono::milliseconds Ts, State x) const override
	{
		(void)Ts; // Suppress compiler warning of unused variables
		size_t n = x.rows();
		return Mat::Identity(n, n);
	}

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	Mat Q(std::chrono::milliseconds Ts, State x) const override
	{
		(void)x;
		(void)Ts;
		return Q_matrix;
	}

private:
	const Mat Q_matrix;
};

} // End namespace Model