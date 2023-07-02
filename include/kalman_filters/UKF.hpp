#pragma once
#include <kalman_filters/Kalman_filter_base.hpp>
#include <models/Model_base.hpp>
#include <math.hpp>

namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class UKF : public Kalman_filter_base<n_x,n_y,n_u,n_v,n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	static constexpr int n_a = n_x+n_v+n_w; // Size of augmented state
	using Mat_aa  = Matrix<double,n_a,n_a>;
	using State_a = Vector<double,n_a>;

	UKF(Models::Model_base<n_x,n_y,n_u,n_v,n_w> *model, State x0, Mat_xx P0) : Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>(x0, P0) 
	{
		// Parameters used for scaling factor and weights W_x0, W_c0 and W_xi
		int alpha{1};
		int beta{0};
		int gamma{0};
		int kappa{3-n_x};
		int lambda{};
		// lambda selected according to the scaled unscented transform. (van der Merwe (2004)) 
		lambda = alpha^2*
		scaling_factor = math::sqrt(n_xx+lambda)
	}

private:
    Model_base<n_x,n_y,n_u,n_v,n_w>* model;
	double scaling_factor;
	double W_x0;
	double W_c0;
	double W_xi;

	Matrix<double,n_a,2*n_a+1> get_sigma_points(State x, Mat_xx P, Mat_vv Q, Mat_ww R)
	{	
		// Make augmented covariance matrix
		Mat_aa P_a;
		P_a << 	P			  , Mat_xv::Zero(), Mat_xw::Zero(),
			  	Mat_vx::Zero(), Q			  , Mat_vw::Zero(),
			  	Mat_wx::Zero(), Mat_wv::Zero(), R			  ;
		
		Mat_aa sqrt_P_a = P_a.llt();

		// Make augmented state vector
		State_a x_a;
		x_a << x, Disturbance::Zero(), Noise::Zero();

		// Calculate sigma points
		Matrix<double,n_a,2*n_a+1> sigma_points;

		// Use the symmetric sigma point set
		sigma_points.row(0) = x_a;
		for (size_t i{1}; i<=n_a; i++)
		{
			sigma_points.row(i)     = x_a + sqrt_P_a.row(i-1);
			sigma_points.row(i+n_a) = x_a - sqrt_P_a.row(i-1);
		}
		return sigma_points;
	}

	Matrix<double,n_x,2*n_a+1> propagate_sigma_points(Matrix<double,n_a,2*n_a+1> sigma_points, Timestep Ts, Input u)
	{
		Matrix<double,n_x,2*n_a+1> propagated_sigma_points;
		for (size_t i{0}; i<2*n_a+1; i++)
		{
			propagated_sigma_points.row(i) = model->f(Ts, sigma_points.block(0,i,n_x,i), u, sigma_points.block(n_x,i,n_x+n_v,i));
		}
		return propagated_sigma_points;
	}

	State next_state(Timestep Ts, Measurement y, Input u = Input::Zero(), Disturbance v = Disturbance::Zero(), Noise w = Noise::Zero()) override final
	{
		Matrix<double,n_a,2*n_a+1> sigma_points = get_sigma_points(this->_x, this->_P_xx, model->Q(Ts, this->_x), model->R(Ts, this->_x));
		Matrix<double,n_x,2*n_a+1> propagated_sigma_points = propagate_sigma_points(sigma_points, Ts, u);
	}

};
}