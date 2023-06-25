#pragma once
#include <models/model_definitions.hpp>
#include <models/EKF_model_base.hpp>

namespace Models {
class LTI_model : public EKF_model_base {
public:
    LTI_model(Mat A, Mat B, Mat C, Mat G, Mat Q, Mat R) : EKF_model_base(), A{A}, B{B}, C{C}, G{G}, Q_mat{Q}, R_mat{R} {}

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
	virtual Mat F_x(Timestep Ts, State x, Input u, Disturbance v) override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		return A;
	}
	virtual Mat H_x(Timestep Ts, State x, Noise w) override final
	{
		(void)Ts;
		(void)x;
		(void)w;
		return C;
	}
	virtual Mat Q(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return Q_mat;
	}
	virtual Mat R(Timestep Ts, State x) override final
	{
		(void)Ts;
		(void)x;
		return R_mat;
	}

	const Mat A;
	const Mat B;
	const Mat C;
	const Mat G;
	const Mat Q_mat;
	const Mat R_mat;
};
}