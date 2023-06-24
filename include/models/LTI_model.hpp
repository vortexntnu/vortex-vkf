#pragma once
#include <models/model_definitions.hpp>
#include <models/EKF_model_base.hpp>

namespace Models {
class LTI_model : public EKF_model_base {
public:
    LTI_model(Mat A, Mat B, Mat C, Mat D, Mat Q, Mat R) : EKF_model_base(), A{A}, B{B}, C{C}, D{D}, Q{Q}, R{R} {}

	virtual State f(Timestep Ts, State x) override
	{
		(void)Ts;
		return A*x;
	}
	virtual Measurement h(Timestep Ts, State x) override
	{
		(void)Ts;
		return C*x;
	}
	virtual Mat F(Timestep Ts, State x) override
	{
		(void)Ts;
		size_t n = x.rows();
		return Eigen::MatrixXd::Identity(n,n);
	}
	virtual Mat H(Timestep Ts, State x) override
	{
		(void)Ts;
		size_t n = x.rows();
		return Eigen::MatrixXd::Identity(n,n);
	}

	const Mat A;
	const Mat B;
	const Mat C;
	const Mat D;
	const Mat Q;
	const Mat R;
};
}