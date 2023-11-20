/**
 * @file erk_methods.hpp
 * @author Eirik Kolås
 * @brief File contains a collection of explicit Runge-Kutta methods
 * @version 0.1
 * @date 2023-11-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <functional>


namespace vortex {
namespace integrator {

/** 
 * @brief Forward Euler method
 * @tparam n_dim_x Dimension of the state vector
 */
template <int n_dim_x> 
class Forward_Euler {
public:
	using Vec_x = Eigen::Vector<double, n_dim_x>;
	using Dyn_mod_func = std::function<Vec_x(double t0, const Vec_x&x0)>;

	static Vec_x integrate(Dyn_mod_func f, double dt, const Vec_x &x0, double t0 = 0.0)
	{
		Vec_x k1 = f(t0, x0);
		return x0 + dt * k1;
	}
};
template <typename DynModT> using Forward_Euler_M = Forward_Euler<DynModT::N_DIM_x>;


/**
 * @brief Runge-Kutta 4th order method
 * @tparam n_dim_x Dimension of the state vector
 */
template <int n_dim_x> 
class RK4 {
public:
	using Vec_x = Eigen::Vector<double, n_dim_x>;
	using Dyn_mod_func = std::function<Vec_x(double t0, const Vec_x&x0)>;

	/**
	 * @brief Integrate the function f over one step dt
	 * 
	 * @param f Function to integrate, must be of the form Vec_x(double dt, const Vec_x&x_k, double t_k=0)
	 * @param dt Time step
	 * @param x0 Start state
	 * @param t0 Start time (optional)
	 * @return Vec_x Integrated state
	 */
	static Vec_x integrate(Dyn_mod_func f, double dt, const Vec_x &x0, double t0 = 0.0)
	{
		Vec_x k1 = f(t0 + 0.0 * dt, x0);
		Vec_x k2 = f(t0 + 0.5 * dt, x0 + 0.5 * dt * k1);
		Vec_x k3 = f(t0 + 0.5 * dt, x0 + 0.5 * dt * k2);
		Vec_x k4 = f(t0 + 1.0 * dt, x0 + 1.0 * dt * k3);

		return x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
	}
};
template <typename DynModT> using RK4_M = RK4<DynModT::N_DIM_x>;


/** 
 * @brief Create an arbitrary explicit Runge-Kutta method from a Butcher table
 * @tparam n_x Dimension of the state vector
 * @tparam n_stages Number of stages in the method
 */
template <int n_dim_x, int n_stages> class Butcher {
public:
	using Vec_x = Eigen::Vector<double, n_dim_x>;
	using Dyn_mod_func = std::function<Vec_x(double t0, const Vec_x&x0)>;

	using Vec_n = Eigen::Vector<double, n_stages>;
	using Mat_nn = Eigen::Matrix<double, n_stages, n_stages>;
	using Mat_xn = Eigen::Matrix<double, n_dim_x, n_stages>;
	/**
	 * @brief Construct an ERK method from a Butcher table. Cannot be passed as a class to a template
	 * as it needs the non-static members A, b and c to be intitalized in an object.
	 * @param A
	 * @param b
	 * @param c
	 */
	Butcher(Mat_nn A, Vec_n b, Vec_n c) : _A{A}, _b{b}, _c{c}
	{
		// Check if Butcher table is valid
		if (A.isLowerTriangular() == false) {
			throw std::invalid_argument("Butcher table: A is not lower triangular");
		}

		// Check if Butcher table is consistent
		if (b.sum() != 1) {
			throw std::invalid_argument("Butcher table: b does not sum to one");
		}
		for (int i = 0; i < A.rows(); i++) {
			if (std::abs(A.row(i).sum() - c(i)) > 1e-6) {
				throw std::invalid_argument("Butcher table: row " + std::to_string(i) + " in A does not sum to c(" + std::to_string(i) + ")");
			}
		}
	}

	Vec_x integrate(Dyn_mod_func f, double dt, const Vec_x &x0, double t0 = 0.0)
	{
		Mat_xn k = Mat_xn::Zero();
		for (size_t i = 0; i < n_stages; i++) {
			k.col(i) = f(t0 + _c(i) * dt, x0 + dt * k * _A.row(i).transpose());
		}
		return x0 + dt * k * _b;
	}

private:
	Mat_nn _A;
	Vec_n _b;
	Vec_n _c;
};

template <int n_dim_x> 
class ODE45 {
public:
	static constexpr int n_stages = 7;
	static constexpr size_t q     = 5; // Order of the method
	
	using Vec_x = Eigen::Vector<double, n_dim_x>;
	using Dyn_mod_func = std::function<Vec_x(double t0, const Vec_x&x0)>;

	using Vec_n = Eigen::Vector<double, n_stages>;
	using Mat_nn = Eigen::Matrix<double, n_stages, n_stages>;
	using Mat_xn = Eigen::Matrix<double, n_dim_x, n_stages>;

	/**
	 * @brief Variable step size Runge-Kutta method
	 *
	 * @param abs_error absolute error. Specify a value for all state variables
	 * @param rel_error relative error. Specify a value for all state variables
	 * @param max_iterations Maximum number of iterations before giving up
	 * @param min_step_size Minimum valid value for the step size in the integration
	 */
	ODE45(double abs_error = 1e-6, double rel_error = 1e-6, size_t max_iterations = 1000, double min_step_size = 1e-9)
	    : ODE45(Vec_x::Ones() * abs_error, Vec_x::Ones() * rel_error, max_iterations, min_step_size)
	{
	}

	/**
	 * @brief Variable step size Runge-Kutta method
	 *
	 * @param abs_error_vec absolute error vector. Specify a value for each state variable
	 * @param rel_error_vec relative error vector. Specify a value for each state variable
	 * @param max_iterations Maximum number of iterations before giving up
	 * @param min_step_size Minimum valid value for the step size in the integration
	 */
	ODE45(Vec_x abs_error_vec = Vec_x::Ones() * 1e-6, Vec_x rel_error_vec = Vec_x::Ones() * 1e-6, size_t max_iterations = 1000, double min_step_size = 1e-9)
	    : _Atol{abs_error_vec}, _Rtol{rel_error_vec}, _max_iter{max_iterations}, _min_step_size{min_step_size}
	{
		// Use the Dormand Prince (RKDP) method
		// clang-format off
		_A << 0           ,  0           , 0           ,  0        ,  0             , 0         , 0,
			  1/5.0       ,  0           , 0           ,  0        ,  0             , 0         , 0,
			  3/40.0      ,  9/40.0      , 0           ,  0        ,  0             , 0         , 0,
			  44/45.0     , -56/15.0     , 32/9.0      ,  0        ,  0             , 0         , 0,
			  19372/6561.0, -25360/2187.0, 64448/6561.0, -212/729.0,  0             , 0         , 0,
			  9017/3168.0 , -355/33.0    , 46732/5247.0,  49/176.0 , -5103/18656.0  , 0         , 0,
			  35/384.0    ,  0           , 500/1113.0  ,  125/192.0, -2187/6784.0   , 11/84.0   , 0;

		Eigen::Matrix<double,2,n_stages> b_T;
		b_T <<35/384.0    ,  0           , 500/1113.0  ,  125/192.0, -2187/6784.0   , 11/84.0   , 0,      // Error of order O(dt^5)
			  5179/57600.0,  0           , 7571/16695.0,  393/640.0, -92097/339200.0, 187/2100.0, 1/40.0; // Error of order O(dt^4)
		_b = b_T.transpose();
		
		_c << 0           ,  1/5.0       , 3/10.0      ,  4/5.0    ,  8/9.0         , 1         , 1;
		// clang-format on
	}

	/**
	 * @brief Integrate the function f over one step dt
	 *
	 * @param f Function to integrate
	 * @param dt Time step size / Simulation time
	 * @param x_k Start state
	 * @param t_k Start time (optional)
	 * @return Integrated state
	 * @throws std::runtime_error if the maximum number of iterations is reached
	 * @throws std::runtime_error if the step size becomes too small
	 */
	Vec_x integrate(Dyn_mod_func f, double dt, const Vec_x &x0, double t0 = 0.0)
	{
		// Copy t and x
		const double t0pdt = t0 + dt; // Final time t0+dt
		double t_i         = t0;      // Intermediate times
		Vec_x x_i          = x0;      // Intermediate states
		double h           = dt;      // Variable step size

		for (size_t iter = 0; iter < _max_iter; iter++) {
			// Compute k_i
			Mat_xn k = Mat_xn::Zero();
			for (size_t i = 0; i < n_stages; i++) {
				k.col(i) = f(t_i + _c(i) * h, x_i + h * k * _A.row(i).transpose());
			}

			Vec_x x_ip1_hat = x_i + h * k * _b.col(0); // Higher order solution
			Vec_x x_ip1     = x_i + h * k * _b.col(1); // Lower order solution

			// From https://scicomp.stackexchange.com/questions/32563/dormand-prince-54-how-to-update-the-stepsize-and-make-accept-reject-decision

			// Compute max(|x_i|, |x_i+1|)
			Vec_x abs_x_i   = x_i.cwiseAbs();
			Vec_x abs_x_ip1 = x_ip1.cwiseAbs(); // Using lower order solution
			Vec_x max_x     = abs_x_i.cwiseMax(abs_x_ip1);

			// Scaling factor
			Vec_x sc = _Atol + max_x.cwiseProduct(_Rtol);

			// Estimate error
			Vec_x temp_error = (x_ip1 - x_ip1_hat).cwiseQuotient(sc);
			double error     = std::sqrt(temp_error.dot(temp_error) / n_dim_x);

			// Accecpt step if error is within tolerance
			if (error < 1) {
				t_i += h;
				x_i = x_ip1_hat;

				if (t_i >= t0pdt) {
					return x_i; // Simulation completed successfully
				}
			}

			// Compute new step size
			double h_new;
			static constexpr double f_ac = std::pow(0.25, 1.0 / (q + 1));                 // safety factor
			h_new                        = f_ac * h * std::pow(1 / error, 1.0 / (q + 1)); // optimal step size
			h_new                        = std::max(h_new, 0.2 * h);                      // limit step size decrease
			h_new                        = std::min(h_new, 5.0 * h);                      // limit step size increase
			h_new                        = std::min(h_new, t0pdt - t_i);                  // limit step size to not overshoot t_kp1
			h                            = h_new;

			if (h < _min_step_size) {
				throw std::runtime_error("Could not reach tolerance within minimum step size" + std::to_string(_min_step_size));
			}
		}
		throw std::runtime_error("Could not reach tolerance within maximum number of iterations " + std::to_string(_max_iter));
	}

private:

	Eigen::Matrix<double, n_stages, n_stages> _A;
	Eigen::Matrix<double, n_stages, 2> _b;
	Eigen::Vector<double, n_stages> _c;

	Vec_x _Atol;
	Vec_x _Rtol;
	size_t _max_iter;
	double _min_step_size;
};

} // namespace integrator
} // namespace vortex