#pragma once
#include <models/model_definitions.hpp>
#include <models/EKF_model_base.hpp>

namespace Models {

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class LTI_model : public EKF_model_base<n_x, n_y, n_u, n_v, n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

    LTI_model(Mat_xx A, Mat_xu B, Mat_yx C, Mat_yu D, Mat_vv Q, Mat_ww R, Mat_xv G, Mat_yw H) : EKF_model_base<n_x,n_y,n_u,n_v,n_w>(), _A{A}, _B{B}, _C{C}, _D{D}, _Q{Q}, _R{R}, _G{G}, _H{H} {}
    LTI_model(Mat_xx A, Mat_xu B, Mat_yx C, Mat_vv Q, Mat_ww R) : LTI_model(A, B, C, Mat_yu::Zero(), Q, R, Mat_xv::Identity(), Mat_yw::Identity()) {}
	LTI_model() : EKF_model_base<n_x,n_y,n_u,n_v,n_w>() {};

	virtual State f(Timestep Ts, State x, Input u = Input::Zero(), Disturbance v = Disturbance::Zero()) override final
	{
		(void)Ts;
		return _A*x + _B*u + _G*v;
	}
	virtual Measurement h(Timestep Ts, State x, Input u = Input::Zero(), Noise w = Noise::Zero()) override final
	{
		(void)Ts;
		return _C*x + _D*u + _H*w;
	}
	virtual Mat_xx F_x(Timestep Ts, State x, Input u, Disturbance v = Disturbance::Zero()) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return _A;
	}
	virtual Mat_xv F_v(Timestep Ts, State x, Input u, Disturbance v = Disturbance::Zero()) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return _G;
	}
	virtual Mat_yx H_x(Timestep Ts, State x, Input u = Input::Zero(), Noise w = Noise::Zero()) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)w;
		return _C;
	}
	virtual Mat_yw H_w(Timestep Ts, State x, Input u = Input::Zero(), Noise w = Noise::Zero()) override final
	{
		(void)x;
		(void)u;
		(void)Ts;
		(void)w;
		return _H;
	}
	virtual Mat_vv Q(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return _Q;
	}
	virtual Mat_ww R(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return _R;
	}

	Mat_xx _A;
	Mat_xu _B;
	Mat_yx _C;
	Mat_yu _D;
	Mat_vv _Q;
	Mat_yy _R;
	Mat_xv _G;
	Mat_yw _H;
};
}
