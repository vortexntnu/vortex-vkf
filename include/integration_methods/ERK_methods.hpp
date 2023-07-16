#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>
#include <functional>
namespace Integrator {
using namespace Models;


// Base class RK_method for RK4 and forward_euler to derive from
template<int n_x, int n_u, int n_v>
class RK_method {
public:
	RK_method() = default;
	virtual ~RK_method() = default;
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	virtual State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) = 0;

};

template<int n_x, int n_u, int n_v>
class None : public RK_method<n_x,n_u,n_v> {
public:
	/**
	 * @brief Does not integrate, just returns f(x_k, u_k, v_k). Use if f is a discrete model
	 * 
	 */
	None() = default;
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) override
	{
		return f(t_k, x_k, u_k, v_k);
	}
};

template<int n_x, int n_u, int n_v>
class RK4_method : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) override
	{
		State k1 = f(0.0*t_k, x_k             , u_k, v_k);
		State k2 = f(0.5*t_k, x_k+0.5*dt/1s*k1, u_k, v_k);
		State k3 = f(0.5*t_k, x_k+0.5*dt/1s*k2, u_k, v_k);
		State k4 = f(1.0*t_k, x_k+1.0*dt/1s*k3, u_k, v_k);

		return x_k + (dt/1s/6.0)*(k1+2*k2+2*k3+k4);
	}
};

template<int n_x, int n_u, int n_v>
class Forward_Euler : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) override
	{
		return x_k + dt/1s*f(t_k, x_k, u_k, v_k);
	}
};

template<int n_x, int n_u, int n_v>
class Heun : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) override
	{
		State k1 = f(0.0*t_k, x_k             , u_k, v_k);
		State k2 = f(1.0*t_k, x_k+1.0*dt/1s*k1, u_k, v_k);

		return x_k + (dt/1s/2.0)*(k1+k2);
	}
};

template<int n_x, int n_u, int n_v>
class Midpoint : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) override
	{
		State k1 = f(0.0*t_k, x_k             , u_k, v_k);
		State k2 = f(1.0*t_k, x_k+1.0*dt/1s*k1, u_k, v_k);

		return x_k + dt/1s*k2;
	}
};



template<int n_x, int n_u, int n_v, int n_stages>
class ERK_method : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	using Function_f = std::function<State(Time t, State x, Input u, Disturbance v)>;
	/**
	 * @brief Construct a ERK method from a Butcher table
	 * 
	 * @param A 
	 * @param b 
	 * @param c 
	 */
	ERK_method(Eigen::Matrix<double,n_stages,n_stages> A, Eigen::Vector<double,n_stages> b, Eigen::Vector<double,n_stages> c) : _A(A), _b(b), _c(c) 
	{
		// Check if Butcher table is valid
		if (A.isLowerTriangular() == false) { throw std::invalid_argument("Butcher table: A is not lower triangular"); }

		// Check if Butcher table is consistent
		if (b.sum() != 1) 					{ throw std::invalid_argument("Butcher table: b does not sum to one"); }
		for (int i = 0; i < A.rows(); i++) 
		{
			if (A.row(i).sum() != c(i)) 	{ throw std::invalid_argument("Butcher table: A row " + std::to_string(i) + " does not sum to c(" + std::to_string(i) + ")"); }
		}
	}

	State integrate(Function_f f, Timestep dt, Time t_k, State x_k, Input u_k, Disturbance v_k) override
	{
		Eigen::Matrix<double,n_x,n_stages> k = Eigen::Matrix<double,n_x,n_stages>::Zero();
		k.col(0) = f(t_k, x_k, u_k, v_k);
		for (size_t i = 1; i < n_stages; i++) 
		{
			for (size_t j = 0; j < i; j++) 
			{
				k.col(i) += f(t_k + _c(i)*dt, x_k + dt/1s*_A(i,j)*k.col(j), u_k, v_k);
			}
		}
		return x_k + dt/1s*k*_b;
	}
private:
	Eigen::Matrix<double,n_stages,n_stages> _A;
	Eigen::Vector<double,n_stages> _b;
	Eigen::Vector<double,n_stages> _c;
};
}