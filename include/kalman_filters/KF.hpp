#pragma once
#include <models/LTI_model.hpp>
#include <kalman_filters/Kalman_filter_base.hpp>

namespace Filters {
using namespace Models;

class KF : public Kalman_filter_base {
public:
	KF(LTI_model *lti_model, State x0, Mat P0) : Kalman_filter_base(x0, P0), model{lti_model} {}
	virtual State next_state(Timestep Ts, Measurement y, Input u, Disturbance v, Noise w) override final
	{
        Mat A = model->A;
        Mat B = model->B;
        Mat C = model->C;
        Mat G = model->G;
        Mat Q = model->Q;
        Mat R = model->R;

		// Predicted State Estimate x_k-
		State x_pred = A*x + B*u + G*v;
		// Predicted State Covariance P_xx
		Mat P_xx_pred = A * P_xx * A.transpose() + C * Q * C.transpose();
		// Predicted Output y_pred
		Measurement y_pred = C*x + w;

		// Output Covariance P_yy
		Mat P_yy = C * P_xx * C.transpose() + R;
		// Cross Covariance P_xy
		Mat P_xy = P_xx * C;

		// Kalman gain K
		size_t m     = P_yy.rows();
		Mat I_m      = Eigen::MatrixXf::Identity(m, m);
		Mat P_yy_inv = P_yy.llt().solve(I_m); // Use Cholesky decomposition for inverting P_yy
		Mat K        = P_xy * P_yy_inv;

		// Corrected State Estimate x_next
		State x_next = x_pred + K * (y - y_pred);
		// Corrected State Covariance P_xx_next
		size_t n      = P_xx.rows();
		Mat I_n       = Eigen::MatrixXf::Identity(n, n);
		Mat P_xx_next = (Eigen::MatrixXd::Identity() - K * C) * P_xx;

		// Update local state
		x    = x_next;
		P_xx = P_xx_next;

		return x_next;
    }
protected:
    LTI_model* model;
};
}