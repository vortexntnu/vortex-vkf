#pragma once
#include <models/model_definitions.hpp>
#include <models/EKF_model_base.hpp>

namespace Models {

template<int n_x, int n_y, int n_u, int n_v, int n_w>
class LTI_model : public EKF_model_base<n_x, n_y, n_u, n_v, n_w> {
public:
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::State;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Measurement;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Input;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Disturbance;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Noise;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_xx; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yy; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_vv; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_ww; 
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_xv;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_xu;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yx;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yu;
	using typename Model_base<n_x,n_y,n_u,n_v,n_w>::Mat_yw;

    LTI_model(Mat_xx A, Mat_xu B, Mat_yx C, Mat_yu D, Mat_vv Q, Mat_ww R, Mat_xv G, Mat_yw H) : EKF_model_base<n_x,n_y,n_u,n_v,n_w>(), A{A}, B{B}, C{C}, D{D}, Q_mat{Q}, R_mat{R}, G{G}, H{H} {}
    LTI_model(Mat_xx A, Mat_xu B, Mat_yx C, Mat_vv Q, Mat_ww R) : LTI_model(A, B, C, Mat_yu::Zero(), Q, R, Mat_xv::Identity(), Mat_yw::Identity()) {}

	virtual State f(Timestep Ts, State x, Input u, Disturbance v) override final
	{
		(void)Ts;
		return A*x + B*u + G*v;
	}
	virtual Measurement h(Timestep Ts, State x, Input u, Noise w) override final
	{
		(void)Ts;
		return C*x + D*u + H*w;
	}
	virtual Mat_xx F_x(Timestep Ts, State x, Input u, Disturbance v) override final
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
	virtual Mat_yx H_x(Timestep Ts, State x, Input u, Noise w) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)w;
		return C;
	}
	virtual Mat_yw H_w(Timestep Ts, State x, Input u, Noise w) override final
	{
		(void)x;
		(void)u;
		(void)Ts;
		(void)w;
		return H;
	}
	virtual Mat_vv Q(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return Q_mat;
	}
	virtual Mat_ww R(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return R_mat;
	}

	const Mat_xx A;
	const Mat_xu B;
	const Mat_yx C;
	const Mat_yu D;
	const Mat_vv Q_mat;
	const Mat_yy R_mat;
	const Mat_xv G;
	const Mat_yw H;
};
template<int n_x, int n_y, int n_u>
using LTI_model2 = LTI_model<n_x, n_y, n_u, n_x, n_y>;
}
