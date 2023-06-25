#pragma once
#include <kalman_filters/Kalman_filter_base.hpp>
#include <models/EKF_model_base.hpp>

namespace Filters {
using namespace Models;

class EKF : public Kalman_filter_base {
public:
	EKF(Models::EKF_model_base *ekf_model, State x0, Mat P0) : Kalman_filter_base(x0, P0) {}

	virtual State next_state(Timestep Ts, Measurement y, Input u, Disturbance v, Noise w) override final
	{
		// Calculate Jacobians
		Mat F_x = model->F_x(Ts, x, u, v);
		Mat F_v = model->F_v(Ts, x, u, v);
		Mat Q = model->Q(Ts, x);
		// Predicted State Estimate x_k-
		State x_pred = model->F_x(Ts, x, u, v);
		// Predicted State Covariance P_xx
		Mat P_xx_pred = F_x * P_xx * F_x.transpose() + F_v * Q * F_v.transpose();
		// Predicted Output y_pred
		Measurement y_pred = model->h(Ts, x_pred, w);

		// Calculate Jacobians
		Mat H_x = model->H_x(Ts, x, w);
		Mat H_w = model->H_w(Ts, x, w);
		Mat R = model->R(Ts, x);
		// Output Covariance P_yy
		Mat P_yy = H_x * P_xx * H_x.transpose() + H_w * R * H_w.transpose();
		// Cross Covariance P_xy
		Mat P_xy = P_xx * H_x;

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
		Mat P_xx_next = (Eigen::MatrixXd::Identity() - K * H_x) * P_xx;

		// Update local state
		x    = x_next;
		P_xx = P_xx_next;

		return x_next;
	}

protected:
	Models::EKF_model_base *model;
};

}