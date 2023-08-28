#pragma once
#include <filters/Kalman_filter_base.hpp>
#include <models/EKF_models.hpp>

namespace Filters {
using namespace Models;

template <class EKF_Model> class EKF : public Kalman_filter_base<EKF_Model> {
public:
	// These type definitions are needed because of the stupid two-phase lookup for dependent names in templates in C++
	using Base = Kalman_filter_base<EKF_Model>;
	using Base::_n_x; using Base::_n_y; using Base::_n_u; using Base::_n_v; using Base::_n_w; // can be comma separated in C++17
	DEFINE_MODEL_TYPES(_n_x, _n_y, _n_u, _n_v, _n_w)

	EKF(std::shared_ptr<EKF_Model> ekf_model, State &x0, Mat_xx &P0) : Base(x0, P0), model{ekf_model}
	{
	}
	~EKF() {}

	State iterate(Time Ts, const Measurement &y, const Input &u = Input::Zero()) override final
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
		// Normalize quaternions if applicable
		x_next = model->post_state_update(x_next);
		// Corrected State Covariance P_xx_next
		Mat_xx P_xx_next = (Mat_xx::Identity() - K * H_x) * P_xx_pred;

		// Update local state
		this->_x    = x_next;
		this->_P_xx = P_xx_next;

		return x_next;
	}

private:
	std::shared_ptr<EKF_Model> model;
};
} // namespace Filters