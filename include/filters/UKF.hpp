#pragma once
#include <filters/Kalman_filter_base.hpp>
#include <models/Model_base.hpp>
#include <math.h>

namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class UKF : public Kalman_filter_base<n_x,n_y,n_u,n_v,n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	static constexpr int n_a = n_x+n_v+n_w; // Size of augmented state
	using Mat_aa  = Matrix<double,n_a,n_a>;
	using State_a = Vector<double,n_a>;

	UKF(Models::Model_base<n_x,n_y,n_u,n_v,n_w> *model, State x0, Mat_xx P0) : Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>(x0, P0), model{model} 
	{
	}

private:
	// Parameters used for calculating scaling factor _GAMMA and weights W_x0, W_c0 and W_xi
	// lambda selected according to the scaled unscented transform. (van der Merwe (2004)) 
	static constexpr double _ALPHA_SQUARED = 1;
	static constexpr double _BETA   	   = 2;
	static constexpr double _KAPPA 		   = 0;
	static constexpr double _LAMBDA 	   = _ALPHA_SQUARED*(n_x+_KAPPA)-n_x;
	static constexpr double _GAMMA         = sqrt(n_x+_LAMBDA);

	static constexpr double _W_x0 = _LAMBDA/(n_x+_LAMBDA);
	static constexpr double _W_c0 = _LAMBDA/(n_x+_LAMBDA)+(1-_ALPHA_SQUARED+_BETA);
	static constexpr double _W_xi = 1/(2*(n_x+_LAMBDA));
	static constexpr double _W_ci = 1/(2*(n_x+_LAMBDA));

    Model_base<n_x,n_y,n_u,n_v,n_w>* model;

	Matrix<double,n_a,2*n_a+1> get_sigma_points(const State& x, const Mat_xx& P, const Mat_vv &Q, const Mat_ww& R)
	{	
		// // Make augmented covariance matrix
		Mat_aa P_a;
		P_a << 	P			  , Mat_xv::Zero(), Mat_xw::Zero(),
			  	Mat_vx::Zero(), Q			  , Mat_vw::Zero(),
			  	Mat_wx::Zero(), Mat_wv::Zero(), R			  ;
		
		Mat_aa sqrt_P_a = P_a.llt();

		// // Make augmented state vector
		State_a x_a;
		x_a << x, Disturbance::Zero(), Noise::Zero();

		// // Calculate sigma points
		Matrix<double,n_a,2*n_a+1> sigma_points;

		// // Use the symmetric sigma point set
		sigma_points.col(0) = x_a;
		for (size_t i{1}; i<=n_a; i++)
		{
			sigma_points.col(i)     = x_a + _GAMMA*sqrt_P_a.col(i-1);
			sigma_points.col(i+n_a) = x_a - _GAMMA*sqrt_P_a.col(i-1);
		}
		return sigma_points;
	}


	State next_state(Timestep Ts, const Measurement& y, const Input& u = Input::Zero()) override final
	{
		Mat_vv Q = model->Q(Ts,this->_x); 
		Mat_ww R = model->R(Ts,this->_x); 
		Matrix<double,n_a,2*n_a+1> sigma_points = get_sigma_points(this->_x, this->_P_xx, Q, R);

		// Propagate sigma points through f
		Matrix<double,n_x,2*n_a+1> sigma_x_pred;
		for (size_t i{0}; i<2*n_a+1; i++)
		{
			sigma_x_pred.col(i) = model->f(Ts, sigma_points.block(0,i,n_x,i), u, sigma_points.block(n_x,i,n_x+n_v-1,i));
		}

		// Predicted State Estimate x_k-
		State x_pred;
		x_pred = _W_x0*sigma_x_pred.col(0);
		for (size_t i{1}; i<=2*n_x; i++)
		{
			x_pred += _W_xi*sigma_x_pred.col(i);
		}

		// Predicted State Covariance P_xx-
		Mat_xx P_xx_pred;
		P_xx_pred = _W_c0*(sigma_x_pred.col(0)-x_pred)*(sigma_x_pred.col(0)-x_pred).transpose();
		for (size_t i{1}; i<=n_x; i++)
		{
			_W_ci*(sigma_x_pred.col(i)-x_pred)*(sigma_x_pred.col(i)-x_pred).transpose();
		}

		// Propagate sigma points through h
		Matrix<double,n_y,2*n_a+1> sigma_y_pred;
		for (size_t i{0}; i<2*n_a+1; i++)
		{
			sigma_y_pred.col(i) = model->h(Ts, sigma_points.block(0,i,n_x,i), u, sigma_points.block(n_x+n_v,i,n_x+n_v+n_w-1,i));
		}

		// // Predicted Output y_pred
		Measurement y_pred;
		y_pred = _W_x0*sigma_y_pred.col(0);
		for (size_t i{1}; i<=2*n_x; i++)
		{
			y_pred += _W_xi*sigma_y_pred.col(i);
		}		

		// Output Covariance P_yy
		Mat_yy P_yy;
		P_yy = _W_c0*(sigma_y_pred.col(0)-y_pred)*(sigma_y_pred.col(0)-y_pred).transpose();
		for (size_t i{1}; i<=n_x; i++)
		{
			P_yy += _W_ci*(sigma_y_pred.col(i)-y_pred)*(sigma_y_pred.col(i)-y_pred).transpose();
		}

		// Cross Covariance P_xy
		Mat_xy P_xy;
		P_xy = _W_c0*(sigma_x_pred.col(0)-x_pred)*(sigma_y_pred.col(0)-y_pred).transpose();
		for (size_t i{1}; i<=n_x; i++)
		{
			P_xy += _W_ci*(sigma_x_pred.col(i)-x_pred)*(sigma_y_pred.col(i)-y_pred).transpose();
		}


		// Kalman gain K
		Mat_yy P_yy_inv = P_yy.llt().solve(Mat_yy::Identity()); // Use Cholesky decomposition for inverting P_yy
		Mat_xy K        = P_xy * P_yy_inv;

		// Corrected State Estimate x_next
		State x_next = x_pred + K * (y - y_pred);
		// Corrected State Covariance P_xx_next
		Mat_xx P_xx_next = P_xx_pred - K * P_yy * K.transpose();

		// Update local state
		this->_x    = x_next;
		this->_P_xx = P_xx_next;

		return x_next;
	}

};
}