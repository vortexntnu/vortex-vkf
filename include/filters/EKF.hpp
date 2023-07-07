#pragma once
#include <filters/Kalman_filter_base.hpp>
#include <models/EKF_model_base.hpp>

namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class EKF : public Kalman_filter_base<n_x,n_y,n_u,n_v,n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

	EKF(Models::EKF_model_base<n_x,n_y,n_u,n_v,n_w> *ekf_model, State x0, Mat_xx P0) : Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>(x0, P0), model{ekf_model} {}

	State iterate(Timestep Ts, const Measurement& y, const Input& u = Input::Zero()) override final
	{
		// Calculate Jacobians F_x, F_v
		Mat_xx F_x = model->F_x(Ts, this->_x, u);
		Mat_xv F_v = model->F_v(Ts, this->_x, u);
		Mat_vv Q   = model->Q(Ts, this->_x);
		// Predicted State Estimate x_k-
		State x_pred = model->f(Ts, this->_x, u);
		// Predicted State Covariance P_xx-
		Mat_xx P_xx_pred = F_x * this->_P_xx * F_x.transpose() + F_v * Q * F_v.transpose();
		// Predicted Output y_pred
		Measurement y_pred = model->h(Ts, x_pred);

		// Calculate Jacobians H_x, H_w
		Mat_yx H_x = model->H_x(Ts, x_pred, u);
		Mat_yw H_w = model->H_w(Ts, x_pred, u);
		Mat_ww R   = model->R(Ts, x_pred);
		// Output Covariance P_yy
		Mat_yy P_yy = H_x * P_xx_pred * H_x.transpose() + H_w * R * H_w.transpose();
		// Cross Covariance P_xy
		Mat_xy P_xy = P_xx_pred * H_x.transpose();

		// Kalman gain K
		Mat_yy P_yy_inv = P_yy.llt().solve(Mat_yy::Identity()); // Use Cholesky decomposition for inverting P_yy
		Mat_xy K        = P_xy * P_yy_inv;

		// Corrected State Estimate x_next
		State x_next = x_pred + K * (y - y_pred);
		// Corrected State Covariance P_xx_next
		Mat_xx P_xx_next = (Mat_xx::Identity() - K * H_x) * P_xx_pred;

		// Update local state
		this->_x    = x_next;
		this->_P_xx = P_xx_next;

		return x_next;
	}

private:
	Models::EKF_model_base<n_x,n_y,n_u,n_v,n_w> *model;
};

}