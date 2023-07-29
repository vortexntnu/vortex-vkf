#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <functional>
#include <cmath>

#include <models/model_definitions.hpp>

namespace Integrator {
using namespace Models;


// Base class RK_method for RK4 and forward_euler to derive from
template<int n_x, int n_u, int n_v>
class RK_method {
public:
	RK_method() = default;
	virtual ~RK_method() = default;
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	virtual State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) = 0;

};

template<int n_x, int n_u, int n_v>
class None : public RK_method<n_x,n_u,n_v> {
public:
	/**
	 * @brief Does not integrate, just returns f(x_k, u_k, v_k). Use if f is a discrete model
	 */
	None() = default;
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		return f(t_k, x_k, u_k, v_k);
	}
};

template<int n_x, int n_u, int n_v>
class RK4 : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		State k1 = f(t_k + 0.0*dt, x_k             , u_k, v_k);
		State k2 = f(t_k + 0.5*dt, x_k+0.5*dt/1s*k1, u_k, v_k);
		State k3 = f(t_k + 0.5*dt, x_k+0.5*dt/1s*k2, u_k, v_k);
		State k4 = f(t_k + 1.0*dt, x_k+1.0*dt/1s*k3, u_k, v_k);

		return x_k + (dt/1s/6.0)*(k1+2*k2+2*k3+k4);
	}
};

template<int n_x, int n_u, int n_v>
class Forward_Euler : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		return x_k + dt/1s*f(t_k, x_k, u_k, v_k);
	}
};

template<int n_x, int n_u, int n_v>
class Heun : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	/**
	 * @brief Heun's method, aka. explicit trapezoidal method
	 * 
	 */
	Heun() = default;
	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		State k1 = f(t_k     , x_k           , u_k, v_k);
		State k2 = f(t_k + dt, x_k + dt/1s*k1, u_k, v_k);

		return x_k + (dt/1s/2.0)*(k1+k2);
	}
};

template<int n_x, int n_u, int n_v>
class Midpoint : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		State k1 = f(t_k + 0.0*dt, x_k             , u_k, v_k);
		State k2 = f(t_k + 0.5*dt, x_k+0.5*dt/1s*k1, u_k, v_k); 
		return x_k + dt/1s*k2;
	}
};



template<int n_x, int n_u, int n_v, int n_stages>
class Butcher : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	/**
	 * @brief Construct a ERK method from a Butcher table
	 * 
	 * @param A 
	 * @param b 
	 * @param c 
	 */
	Butcher(Eigen::Matrix<double,n_stages,n_stages> A, Eigen::Vector<double,n_stages> b, Eigen::Vector<double,n_stages> c) : _A{A}, _b{b}, _c{c}
	{
		// Check if Butcher table is valid
		if (A.isLowerTriangular() == false) { throw std::invalid_argument("Butcher table: A is not lower triangular"); }

		// Check if Butcher table is consistent
		if (b.sum() != 1) 					{ throw std::invalid_argument("Butcher table: b does not sum to one"); }
		for (int i = 0; i < A.rows(); i++) 
		{
			if (std::abs(A.row(i).sum() - c(i)) > 1e-6) 	
			{ 
				throw std::invalid_argument("Butcher table: row " + std::to_string(i) + " in A does not sum to c(" + std::to_string(i) + ")"); 
			}
		}
	}

	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		Eigen::Matrix<double,n_x,n_stages> k = Eigen::Matrix<double,n_x,n_stages>::Zero();
		for (size_t i = 0; i < n_stages; i++) 
		{
			k.col(i) = f(t_k + _c(i)*dt, x_k + dt/1s*k*_A.row(i).transpose(), u_k, v_k);
		}
		return x_k + dt/1s*k*_b;
	}
private:
	Eigen::Matrix<double,n_stages,n_stages> _A;
	Eigen::Vector<double,n_stages> _b;
	Eigen::Vector<double,n_stages> _c;
};

template<int n_x, int n_u, int n_v>
class ODE45 : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	/**
	 * @brief Variable step size Runge-Kutta method
	 * 
	 * @param abs_error absolute error. Specify a value for all state variables
	 * @param rel_error relative error. Specify a value for all state variables
	 * @param max_iterations Maximum number of iterations before giving up
	 * @param min_step_size Minimum valid value for the step size in the integration
	 */
	ODE45(double abs_error=1e-6, double rel_error=1e-6, size_t max_iterations=1000, Timestep min_step_size=1ns) : ODE45(State::Ones()*abs_error, State::Ones()*rel_error, max_iterations, min_step_size) {}

	/**
	 * @brief Variable step size Runge-Kutta method
	 * 
	 * @param abs_error_vec absolute error vector. Specify a value for each state variable
	 * @param rel_error_vec relative error vector. Specify a value for each state variable
	 * @param max_iterations Maximum number of iterations before giving up
	 * @param min_step_size Minimum valid value for the step size in the integration
	 */
	ODE45(State abs_error_vec=State::Ones()*1e-6, State rel_error_vec=State::Ones()*1e-6, size_t max_iterations=1000, Timestep min_step_size=1ns) : _Atol{abs_error_vec}, _Rtol{rel_error_vec}, _max_iter{max_iterations}, _min_step_size{min_step_size}
	{
		// Use the Dormand Prince (RKDP) method
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

	}


	/**
	 * @brief Integrate the function f over one step dt
	 * 
	 * @param f Function to integrate
	 * @param dt Time step size / Simulation time
	 * @param t_k Start time
	 * @param x_k Start state
	 * @param u_k Input considered constant over the step
	 * @param v_k Disturbance considered constant over the step
	 * @return Integrated state
	 * @throws std::runtime_error if the maximum number of iterations is reached
	 * @throws std::runtime_error if the step size becomes too small
	 */
	State integrate(State_dot f, Timestep dt, Time t_k, const State& x_k, const Input& u_k, const Disturbance& v_k) override
	{
		// Copy t and x
		const Time t_kp1 = t_k + dt;	// Final time t_k+1
		Time t_i = t_k; 				// Intermediate times
		State x_i = x_k; 				// Intermediate states
		Timestep h = dt; 				// Variable step size


		for (size_t iter=0; iter<_max_iter; iter++)
		{
			// Compute k_i
			Eigen::Matrix<double,n_x,n_stages> k = Eigen::Matrix<double,n_x,n_stages>::Zero();
			for (size_t i = 0; i < n_stages; i++) 
			{
				k.col(i) = f(t_i + _c(i)*h, x_i + h/1s*k*_A.row(i).transpose(), u_k, v_k);
			}
			
			State x_ip1_hat = x_i + h/1s*k*_b.col(0); // Higher order solution
			State x_ip1 	= x_i + h/1s*k*_b.col(1); // Lower order solution

			// From https://scicomp.stackexchange.com/questions/32563/dormand-prince-54-how-to-update-the-stepsize-and-make-accept-reject-decision

			// Compute max(|x_i|, |x_i+1|)
			State abs_x_i   = x_i.cwiseAbs();
			State abs_x_ip1 = x_ip1.cwiseAbs(); // Using lower order solution
			State max_x     = abs_x_i.cwiseMax(abs_x_ip1);

			// Scaling factor
			State sc = _Atol + max_x.cwiseProduct(_Rtol);

			// Estimate error
			State temp_error = (x_ip1 - x_ip1_hat).cwiseQuotient(sc);
			double error = std::sqrt(temp_error.dot(temp_error)/n_x);

			// Accecpt step if error is within tolerance
			if (error < 1) 
			{
				t_i += h;
				x_i = x_ip1_hat;

				if (t_i >= t_kp1) 
				{
					return x_i; // Simulation completed successfully
				} 
			}

			// Compute new step size
			Timestep h_new;
			static constexpr double f_ac = std::pow(0.25,1.0/(q+1)); 		// safety factor
			h_new = f_ac*h*std::pow(1/error, 1.0/(q+1)); 					// optimal step size
			h_new = std::max(h_new/1s, 0.2*h/1s)*1s; 						// limit step size decrease
			h_new = std::min(h_new/1s, 5.0*h/1s)*1s; 						// limit step size increase
			h_new = std::min(h_new/1s, (t_kp1 - t_i)/1s)*1s; 				// limit step size to not overshoot t_kp1
			h = h_new;


			if (h < _min_step_size) {
				throw std::runtime_error("Could not reach tolerance within minimum step size" + std::to_string(_min_step_size/1s));
			}
		}
		throw std::runtime_error("Could not reach tolerance within maximum number of iterations " + std::to_string(_max_iter));

	}
private:
	static constexpr int n_stages = 7;
	static constexpr size_t q = 5; // Order of the method

	Eigen::Matrix<double,n_stages,n_stages> _A;
	Eigen::Matrix<double,n_stages,2> _b;
	Eigen::Vector<double,n_stages> _c;

	State _Atol;
	State _Rtol;
	size_t _max_iter;
	Timestep _min_step_size;
};

}