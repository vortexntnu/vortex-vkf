#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>

namespace Model {

template <size_t n> class Measurement_model {
public:
	using Vec = Eigen::Matrix<double, n, 1>;
	using Mat = Eigen::Matrix<double, n, n>;
	/**
	 * @brief Parent class for dynamic models
	 */
	Measurement_model() {}

	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	virtual Vec f(std::chrono::milliseconds Ts, Vec x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(std::chrono::milliseconds Ts, Vec x) const = 0;
};

template <size_t n> class LTV_model : public Measurement_model<n> {
public:
	using typename Measurement_model<n>::Vec;
	using typename Measurement_model<n>::Mat;
	/**
	 * @brief Parent class for Linear time varying models
	 * Instead of defining the system function \p f, define the Jacobian \p F
	 */
	LTV_model() : Measurement_model<n>{} {}
	/**
	 * @brief Discrete prediction equation f:
	 * Calculate the zero noise prediction at time \p Ts from \p x.
	 * @param x State
	 * @param Ts Time-step
	 * @return The next state x_(k+1) = F x_k
	 */
	Vec f(std::chrono::milliseconds Ts, Vec x) const { return F(Ts, x) * x; }

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	virtual Mat F(std::chrono::milliseconds Ts, Vec x) const = 0;

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return System noise covariance matrix Q
	 */
	virtual Mat Q(std::chrono::milliseconds Ts, Vec x) const = 0;
};

constexpr size_t LM_size{7}; // Size of the landmark model (x, y, z and quaternion)
class Landmark : public LTV_model<LM_size> {
public:
	/**
	 * @brief Model for stationary objects with x, y, z and quaternion as states
	 * @param Q_weights The weights of the system noise covariance matrix
	 */
	Landmark(Vec Q_weights) : LTV_model{}, Q_matrix{Eigen::DiagonalMatrix<double, LM_size, LM_size>{Q_weights}} {}
	Landmark() : LTV_model{}, Q_matrix{Mat::Identity()} {}

	/**
	 * @brief Jacobian of f:
	 * Calculate the transition function jacobian for time \p Ts at \p x
	 * @param x State
	 * @param Ts Time-step
	 * @return Jacobian F
	 */
	Mat F(std::chrono::milliseconds Ts, Vec x) const
	{
		(void)x; // Suppress compiler warning of unused variables
		(void)Ts;
		return Mat::Identity();
	}

	/**
	 * @brief Covariance matrix of model:
	 * Calculate the transition covariance \p Q for time \p Ts
	 * @param x State
	 * @param Ts Time-step
	 * @return Covariance matrix Q
	 */
	Mat Q(std::chrono::milliseconds Ts, Vec x) const
	{
		(void)x;
		(void)Ts;
		return Q_matrix;
	}

private:
	const Mat Q_matrix;
};

} // End namespace Model