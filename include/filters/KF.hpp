#pragma once
#include <models/LTI_model.hpp>
#include <filters/Kalman_filter_base.hpp>

namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class KF : public Kalman_filter_base<n_x,n_y,n_u,n_v,n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	KF(LTI_model<n_x,n_y,n_u,n_v,n_w> *lti_model, State x0, Mat_xx P0) : Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>(x0, P0), model{lti_model} {}
	State iterate(Timestep Ts, const Measurement& y, const Input& u = Input::Zero()) override final
	{
        Mat_xx A = model->_A;
        Mat_xu B = model->_B;
        Mat_yx C = model->_C;
        Mat_yu D = model->_D;
        Mat_vv Q = model->_Q;
        Mat_yy R = model->_R;
        Mat_xv G = model->_G;
        Mat_yw H = model->_H;

		// Predicted State Estimate x_k-
		State x_pred = A*this->_x + B*u;
		// Predicted State Covariance P_xx
		Mat_xx P_xx_pred = A*this->_P_xx*A.transpose() + G*Q*G.transpose();
		// Predicted Output y_pred
		Measurement y_pred = C*x_pred + D*u;

		// Output Covariance P_yy
		Mat_yy P_yy = C*P_xx_pred*C.transpose() + H*R*H.transpose();
		// Cross Covariance P_xy
		Mat_xy P_xy = P_xx_pred * C;

		// Kalman gain K
		Mat_yy P_yy_inv = P_yy.llt().solve(Mat_yy::Identity()); // Use Cholesky decomposition for inverting P_yy
		Mat_xy K        = P_xy * P_yy_inv;

		// Corrected State Estimate x_next
		State x_next = x_pred + K * (y - y_pred);
		// Corrected State Covariance P_xx_next
		Mat_xx P_xx_next = (Mat_xx::Identity() - K * C) * P_xx_pred;

		// Update local state
		this->_x    = x_next;
		this->_P_xx = P_xx_next;

		return x_next;
    }
private:
    LTI_model<n_x,n_y,n_u,n_v,n_w>* model;
};
}