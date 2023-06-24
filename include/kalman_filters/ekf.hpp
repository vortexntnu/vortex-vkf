#pragma once
#include <kalman_filters/Kalman_filter_base.hpp>
#include <models/EKF_model_base.hpp>
using namespace Models;

class EKF : public Kalman_filter_base {
public:
	EKF(Models::EKF_model_base *ekf_model, State x0, Mat P0) : Kalman_filter_base(x0, P0) {}

	State next_state(Timestep Ts, Measurement y, Input u) override
	{
		// Calculate Jacobians
		Mat F = model->F(Ts, x);
		Mat Q = model->Q(Ts, x);
		// Predicted State Estimate x_k-
		State x_pred = model->F(Ts, x);
		// Predicted State Covariance P_xx
		Mat P_xx_pred = F * P_xx * F.transpose() + Q;
		// Predicted Output y_pred
		Measurement y_pred = model->h(Ts, x_pred);

		// Calculate Jacobians
		Mat H = model->H(Ts, x);
		Mat R = model->R(Ts, x);
		// Output Covariance P_yy
		Mat P_yy = H * P_xx * H.transpose() + R;
		// Cross Covariance P_xy
		Mat P_xy = P_xx * H;

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
		Mat P_xx_next = (Eigen::MatrixXd::Identity() - K * H) * P_xx;

		// Update local state
		x    = x_next;
		P_xx = P_xx_next;

		return x_next;
	}

protected:
	Models::EKF_model_base *model;
};
