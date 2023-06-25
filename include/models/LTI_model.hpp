#pragma once
#include <models/model_definitions.hpp>
#include <models/EKF_model_base.hpp>

namespace Models {
class LTI_model : public EKF_model_base {
public:
    LTI_model(Mat A, Mat B, Mat C, Mat D, Mat Q, Mat R) : EKF_model_base(), A{A}, B{B}, C{C}, D{D}, Q_mat{Q}, R_mat{R} {}

	virtual State f(Timestep Ts, State x) final
	{
		(void)Ts;
		return A*x;
	}
	virtual Measurement h(Timestep Ts, State x) final
	{
		(void)Ts;
		return C*x;
	}
	virtual Mat F(Timestep Ts, State x) final
	{
		(void)Ts;
		(void)x;
		return A;
	}
	virtual Mat H(Timestep Ts, State x) final
	{
		(void)Ts;
		(void)x;
		return C;
	}
	virtual Mat Q(Timestep Ts, State x) final
	{
		(void)Ts;

		return Q_mat;
	}
	virtual Mat R(Timestep Ts, State x) final
	{
		(void)Ts;
		(void)x;
		return R_mat;
	}

	const Mat A;
	const Mat B;
	const Mat C;
	const Mat D;
	const Mat Q_mat;
	const Mat R_mat;
};
}