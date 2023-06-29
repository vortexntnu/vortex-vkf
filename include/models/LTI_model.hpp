#pragma once
#include <models/model_definitions.hpp>
#include <models/EKF_model_base.hpp>

namespace Models {

template<int n_x, int n_y, int n_u, int n_v>
class LTI_model : public EKF_model_base<n_x, n_y, n_u, n_v> {
public:
	using typename Model_base<n_x,n_y,n_u,n_v>::State;
	using typename Model_base<n_x,n_y,n_u,n_v>::Measurement;
	using typename Model_base<n_x,n_y,n_u,n_v>::Input;
	using typename Model_base<n_x,n_y,n_u,n_v>::Disturbance;
	using typename Model_base<n_x,n_y,n_u,n_v>::Noise;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_vv; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_yy; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_vv; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_ww; 
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_xv;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_xu;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_yx;
	using typename Model_base<n_x,n_y,n_u,n_v>::Mat_yw;

    LTI_model(Mat_vv A, Mat_xu B, Mat_yx C, Mat_xv G, Mat_vv Q, Mat_yy R) : EKF_model_base<n_x, n_y, n_u, n_v>(), A{A}, B{B}, C{C}, G{G}, Q_mat{Q}, R_mat{R} {}

	virtual State f(Timestep Ts, State x, Input u, Disturbance v) override final
	{
		(void)Ts;
		return A*x + B*u + G*v;
	}
	virtual Measurement h(Timestep Ts, State x, Noise w) override final
	{
		(void)Ts;
		return C*x + w;
	}
	virtual Mat_vv F_x(Timestep Ts, State x, Input u, Disturbance v) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return A;
	}
	virtual Mat_xv F_v(Timestep Ts, State x, Input u, Disturbance v) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return G;
	}
	virtual Mat_yx H_x(Timestep Ts, State x, Noise w) override final
	{
		(void)Ts;
		(void)x;
		(void)w;
		return C;
	}
	virtual Mat_yw H_w(Timestep Ts, State x, Noise w) override final
	{
		(void)x;
		(void)Ts;
		(void)w;
		return Mat_yy::Identity();
	}
	virtual Mat_vv Q(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return Q_mat;
	}
	virtual Mat_yy R(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return R_mat;
	}

	const Mat_vv A;
	const Mat_xu B;
	const Mat_yx C;
	const Mat_xv G;
	const Mat_vv Q_mat;
	const Mat_yy R_mat;
};
}