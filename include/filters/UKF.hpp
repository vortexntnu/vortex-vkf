#pragma once
#include <cmath>
#include <filters/Kalman_filter_base.hpp>
#include <memory>
// #include <models/Model_base.hpp>

namespace Filters {
using namespace Models;

template <class Model> class UKF : public Kalman_filter_base<Model> {
public:
	// These type definitions are needed because of the stupid two-phase lookup for dependent names in templates in C++
	using Base = Kalman_filter_base<Model>;
	using Base::_n_x; using Base::_n_y; using Base::_n_u; using Base::_n_v; using Base::_n_w; // can be comma separated in C++17
	DEFINE_MODEL_TYPES(_n_x, _n_y, _n_u, _n_v, _n_w)
	static constexpr int _n_a = _n_x + _n_v + _n_w; // Size of augmented state
	using Mat_aa             = Matrix<double, _n_a, _n_a>;
	using State_a            = Vector<double, _n_a>;

	UKF(std::shared_ptr<Model> model, State x0, Mat_xx P0) : Kalman_filter_base<Model>(x0, P0), model{model} {}
	~UKF() {}

private:
	// Parameters used for calculating scaling factor, _GAMMA and weights W_x0, W_c0 and W_xi
	// lambda selected according to the scaled unscented transform. (van der Merwe (2004))
	static constexpr double _ALPHA_SQUARED = 1;
	static constexpr double _BETA          = 2;
	static constexpr double _KAPPA         = 0;
	static constexpr double _LAMBDA        = _ALPHA_SQUARED * (_n_a + _KAPPA) - _n_a;
	static constexpr double _GAMMA         = std::sqrt(_n_a + _LAMBDA);

	static constexpr double _W_x0 = _LAMBDA / (_n_a + _LAMBDA);
	static constexpr double _W_c0 = _LAMBDA / (_n_a + _LAMBDA) + (1 - _ALPHA_SQUARED + _BETA);
	static constexpr double _W_xi = 1 / (2 * (_n_a + _LAMBDA));
	static constexpr double _W_ci = 1 / (2 * (_n_a + _LAMBDA));

	std::shared_ptr<Model> model;

	Matrix<double, _n_a, 2 * _n_a + 1> get_sigma_points(const State &x, const Mat_xx &P, const Mat_vv &Q, const Mat_ww &R)
	{
		// Make augmented covariance matrix
		Mat_aa P_a;
		// clang-format off
		P_a << 	P			  , Mat_xv::Zero(), Mat_xw::Zero(),
			  	Mat_vx::Zero(), Q			  , Mat_vw::Zero(),
			  	Mat_wx::Zero(), Mat_wv::Zero(), R;	
		// clang-format on
		Mat_aa sqrt_P_a = P_a.llt().matrixLLT();

		// Make augmented state vector
		State_a x_a;
		x_a << x, Disturbance::Zero(), Noise::Zero();

		// Calculate sigma points
		Matrix<double, _n_a, 2 * _n_a + 1> sigma_points;

		// Use the symmetric sigma point set
		sigma_points.col(0) = x_a;
		for (size_t i = 1; i <= _n_a; i++) {
			sigma_points.col(i)       = x_a + _GAMMA * sqrt_P_a.col(i - 1);
			sigma_points.col(i + _n_a) = x_a - _GAMMA * sqrt_P_a.col(i - 1);
		}
		return sigma_points;
	}

public:
	State iterate(Time t, const Measurement &y, const Input &u = Input::Zero()) override final
	{
		Mat_vv Q                                     = model->Q(t, this->_x);
		Mat_ww R                                     = model->R(t, this->_x);

		Matrix<double, _n_a, 2 *_n_a + 1> sigma_points = get_sigma_points(this->_x, this->_P_xx, Q, R);

		// Propagate sigma points through f
		Matrix<double, _n_x, 2 * _n_a + 1> sigma_x_pred;
		for (size_t i = 0; i < 2 * _n_a + 1; i++) {
			auto x_i            = sigma_points.template block<_n_x, 1>(0, i);
			auto v_i            = sigma_points.template block<_n_v, 1>(_n_x, i);
			sigma_x_pred.col(i) = model->f(t, x_i, u, v_i);
		}

		// Predicted State Estimate x_k-
		State x_pred;
		x_pred = _W_x0 * sigma_x_pred.col(0);
		for (size_t i = 1; i < 2 * _n_a + 1; i++) {
			x_pred += _W_xi * sigma_x_pred.col(i);
		}

		// Predicted State Covariance P_xx-
		Mat_xx P_xx_pred;
		P_xx_pred = _W_c0 * (sigma_x_pred.col(0) - x_pred) * (sigma_x_pred.col(0) - x_pred).transpose();
		for (size_t i = 1; i < 2 * _n_a + 1; i++) {
			_W_ci *(sigma_x_pred.col(i) - x_pred) * (sigma_x_pred.col(i) - x_pred).transpose();
		}

		// Propagate sigma points through h
		Matrix<double, _n_y, 2 * _n_a + 1> sigma_y_pred;
		for (size_t i = 0; i < 2 * _n_a + 1; i++) {
			auto x_i            = sigma_points.template block<_n_x, 1>(0, i);
			auto w_i            = sigma_points.template block<_n_w, 1>(_n_x + _n_v, i);
			sigma_y_pred.col(i) = model->h(t, x_i, u, w_i);
		}

		// Predicted Output y_pred
		Measurement y_pred;
		y_pred = _W_x0 * sigma_y_pred.col(0);
		for (size_t i = 1; i < 2 * _n_a + 1; i++) {
			y_pred += _W_xi * sigma_y_pred.col(i);
		}

		// Output Covariance P_yy
		Mat_yy P_yy;
		P_yy = _W_c0 * (sigma_y_pred.col(0) - y_pred) * (sigma_y_pred.col(0) - y_pred).transpose();
		for (size_t i = 1; i < 2 * _n_a + 1; i++) {
			P_yy += _W_ci * (sigma_y_pred.col(i) - y_pred) * (sigma_y_pred.col(i) - y_pred).transpose();
		}

		// Cross Covariance P_xy
		Mat_xy P_xy;
		P_xy = _W_c0 * (sigma_x_pred.col(0) - x_pred) * (sigma_y_pred.col(0) - y_pred).transpose();
		for (size_t i = 1; i < 2 * _n_a + 1; i++) {
			P_xy += _W_ci * (sigma_x_pred.col(i) - x_pred) * (sigma_y_pred.col(i) - y_pred).transpose();
		}

		// Kalman gain K
		Mat_yy P_yy_inv = P_yy.llt().solve(Mat_yy::Identity()); // Use Cholesky decomposition for inverting P_yy
		Mat_xy K        = P_xy * P_yy_inv;

		// Corrected State Estimate x_next
		State x_next = x_pred + K * (y - y_pred);
		// Normalize quaternions if applicable
		x_next = model->post_state_update(x_next);
		// Corrected State Covariance P_xx_next
		Mat_xx P_xx_next = P_xx_pred - K * P_yy * K.transpose();

		// Update local state
		this->_x    = x_next;
		this->_P_xx = P_xx_next;

		return x_next;
	}
};


// required namespace-scope declarations to avoid linker errors
template <class Model> constexpr int    UKF<Model>::_n_a;
template <class Model> constexpr double UKF<Model>::_ALPHA_SQUARED;
template <class Model> constexpr double UKF<Model>::_BETA;
template <class Model> constexpr double UKF<Model>::_KAPPA;
template <class Model> constexpr double UKF<Model>::_LAMBDA;
template <class Model> constexpr double UKF<Model>::_GAMMA;
template <class Model> constexpr double UKF<Model>::_W_x0;
template <class Model> constexpr double UKF<Model>::_W_c0;
template <class Model> constexpr double UKF<Model>::_W_xi;
template <class Model> constexpr double UKF<Model>::_W_ci;

} // namespace Filters
