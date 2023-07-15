#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
#include <models/model_definitions.hpp>
namespace Integrator {
using namespace Models;


// Base class RK_method for RK4 and forward_euler to derive from
template<int n_x, int n_u, int n_v>
class RK_method {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	RK_method() = default;
	virtual ~RK_method() = default;
	virtual State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) = 0;

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
	State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) override
	{
		return f(Ts, x_k, u_k, v_k);
	}
};

template<int n_x, int n_u, int n_v>
class RK4_method : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) override
	{
		State k1 = f(0.0*Ts, x_k             , u_k, v_k);
		State k2 = f(0.5*Ts, x_k+0.5*Ts/1s*k1, u_k, v_k);
		State k3 = f(0.5*Ts, x_k+0.5*Ts/1s*k2, u_k, v_k);
		State k4 = f(1.0*Ts, x_k+1.0*Ts/1s*k3, u_k, v_k);

		return x_k + (Ts/1s/6.0)*(k1+2*k2+2*k3+k4);
	}
};

template<int n_x, int n_u, int n_v>
class Forward_Euler : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) override
	{
		return x_k + Ts/1s*f(0.0, x_k, u_k, v_k);
	}
};

template<int n_x, int n_u, int n_v>
class Heun : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) override
	{
		State k1 = f(0.0*Ts, x_k             , u_k, v_k);
		State k2 = f(1.0*Ts, x_k+1.0*Ts/1s*k1, u_k, v_k);

		return x_k + (Ts/1s/2.0)*(k1+k2);
	}
};

template<int n_x, int n_u, int n_v>
class Midpoint : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) override
	{
		State k1 = f(0.0*Ts, x_k             , u_k, v_k);
		State k2 = f(1.0*Ts, x_k+1.0*Ts/1s*k1, u_k, v_k);

		return x_k + Ts/1s*k2;
	}
};



template<int n_x, int n_u, int n_v>
class ERK_method : public RK_method<n_x,n_u,n_v> {
public:
	DEFINE_MODEL_TYPES(n_x,0,n_u,n_v,0)
	ERK_method(const Eigen::MatrixXd& A, const Eigen::RowVectorXd& b, const Eigen::VectorXd& c) : _A(A), _b(b), _c(c) 
	{
		// Check if Butcher table is valid
		if (A.rows() != A.cols()) 			{ throw std::invalid_argument("Butcher table A is not square"); }
		if (A.cols() != b.cols()) 			{ throw std::invalid_argument("Butcher table A and b have different number of columns"); }
		if (A.rows() != c.rows()) 			{ throw std::invalid_argument("Butcher table A and c have different number of rows"); }
		if (A.isLowerTriangular() == false) { throw std::invalid_argument("Butcher table A is not lower triangular"); }

		// Check if Butcher table is consistent
		if (b.sum() != 1) 					{ throw std::invalid_argument("Butcher table b does not sum to one"); }
		for (int i = 0; i < A.rows(); i++) {
			if (A.row(i).sum() != c(i)) 	{ throw std::invalid_argument("Butcher table A row " + std::to_string(i) + " does not sum to c(" + std::to_string(i) + ")"); }
		}
	}
	State integrate(State (*f)(Timestep Ts, State x, Input u, Disturbance v), Timestep Ts, State x_k, Input u_k, Disturbance v_k) override
	{
		Eigen::RowVectorXd k = Eigen::VectorXd::Zero(_b.cols());
		for (int i = 0; i < k.rows(); i++) {
			for (int j = 0; j < i; j++) {
				k(i) += f(_c(i)*Ts, x_k+Ts/1s*_A(i,j)*k(j), u_k, v_k);
			}
		}
		return x_k + Ts/1s*_b*k;
	}
private:
	Eigen::MatrixXd _A;
	Eigen::RowVectorXd _b;
	Eigen::VectorXd _c;
};
}