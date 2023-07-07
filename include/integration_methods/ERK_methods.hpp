#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>
namespace RKmethods {
using Eigen::Matrix;
using Eigen::Vector;
using namespace std::chrono_literals;
using Timestep = std::chrono::milliseconds;
template<int n>
using VectorNd = Vector<double,n>;

template<int n_x, int n_u, int n_v>
VectorNd<n_x> forward_euler(VectorNd<n_x> (*f)(Timestep Ts, VectorNd<n_x> x, VectorNd<n_u> u, VectorNd<n_v> v), Timestep Ts, VectorNd<n_x> x_k, VectorNd<n_u> u_k, VectorNd<n_v> v_k)
{
	return x_k + Ts*f(0.0, x_k, u_k, v_k);
}

template<int n_x, int n_u, int n_v>
VectorNd<n_x> RK4(VectorNd<n_x> (*f)(Timestep Ts, VectorNd<n_x> x, VectorNd<n_u> u, VectorNd<n_v> v), Timestep Ts, VectorNd<n_x> x_k, VectorNd<n_u> u_k, VectorNd<n_v> v_k)
{
	VectorNd<n_x> k1 = f(0.0*Ts, x_k          , u_k, v_k);
	VectorNd<n_x> k2 = f(0.5*Ts, x_k+0.5*Ts*k1, u_k, v_k);
	VectorNd<n_x> k3 = f(0.5*Ts, x_k+0.5*Ts*k2, u_k, v_k);
	VectorNd<n_x> k4 = f(1.0*Ts, x_k+1.0*Ts*k3, u_k, v_k);

	return x_k + Ts/6.0*(k1+2*k2+2*k3+k4);
}
}