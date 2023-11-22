#include <gtest/gtest.h>
#include <iostream>

#include "test_models.hpp"
#include <gnuplot-iostream.h>
#include <random>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

class EKFTestCVModel : public ::testing::Test {
protected:
	using PosMeasModel = vortex::models::IdentitySensorModel<4, 2>;
	using CVModel      = vortex::models::ConstantVelocity;
	using Vec_x        = typename CVModel::Vec_x;
	using Vec_u        = typename CVModel::Vec_u;
	using Mat_xx       = typename CVModel::Mat_xx;
	using Gauss_x      = typename CVModel::Gauss_x;
	using Gauss_z      = typename PosMeasModel::Gauss_z;
	using Vec_z        = typename PosMeasModel::Vec_z;

	void SetUp() override
	{
		// Create dynamic model
		dynamic_model_ = std::make_shared<CVModel>(1e-3);
		// Create sensor model
		sensor_model_ = std::make_shared<PosMeasModel>(1e-2);
		// Create EKF
		ekf_ = std::make_shared<vortex::filter::EKF_M<CVModel, PosMeasModel>>(dynamic_model_, sensor_model_);
	}

	std::shared_ptr<CVModel> dynamic_model_;
	std::shared_ptr<PosMeasModel> sensor_model_;
	std::shared_ptr<vortex::filter::EKF_M<CVModel, PosMeasModel>> ekf_;
};

TEST_F(EKFTestCVModel, Predict)
{
	// Initial state
	Gauss_x x({0, 0, 1, 0}, Mat_xx::Identity());
	double dt = 0.1;
	// Predict
	auto pred          = ekf_->predict(dt, x);
	Gauss_x x_est_pred = pred.first;
	Gauss_z z_est_pred = pred.second;

	Vec_x x_true = x.mean() + Vec_x({dt, 0, 0, 0});
	Vec_z z_true = x_true.head(2);
	ASSERT_EQ(x_est_pred.mean(), x_true);
	ASSERT_EQ(z_est_pred.mean(), z_true);
}

TEST_F(EKFTestCVModel, Convergence)
{
	// Random number generator
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d_disturbance{0, 1e-3};
	std::normal_distribution<> d_noise{0, 1e-2};

	// Initial state
	Gauss_x x0({0, 0, 0.5, 0}, Mat_xx::Identity());
	double dt = 0.1;

	std::vector<double> time;
	std::vector<Vec_x> x_true;
	std::vector<Gauss_x> x_est;
	std::vector<Vec_z> z_meas;
	std::vector<Gauss_z> z_est;

	// Simulate
	time.push_back(0);
	x_true.push_back(x0.mean());
	x_est.push_back(x0);
	for (int i = 0; i < 100; i++) {
		// Simulate
		Vec_x x_true_i = dynamic_model_->sample_f_d(dt, x_true.back(), Vec_u::Zero(), gen);
		Vec_z z_meas_i = sensor_model_->sample_h(x_true_i, gen);
		x_true.push_back(x_true_i);
		z_meas.push_back(z_meas_i);

		// Predict and update
		auto step          = ekf_->step(dt, x_est.back(), z_meas_i);
		Gauss_x x_est_upd  = std::get<0>(step);
		Gauss_z z_est_pred = std::get<2>(step);

		// Save results
		time.push_back(time.back() + dt);
		x_est.push_back(x_est_upd);
		z_est.push_back(z_est_pred);
	}

	// Test that the state converges to the true state
	ASSERT_NEAR(x_est.back().mean()(0), x_true.back()(0), 1e-1);
	ASSERT_NEAR(x_est.back().mean()(1), x_true.back()(1), 1e-1);
	ASSERT_NEAR(x_est.back().mean()(2), x_true.back()(2), 1e-1);
	ASSERT_NEAR(x_est.back().mean()(3), x_true.back()(3), 1e-1);

	// Plot the results
	std::vector<double> x_true_x, x_true_y, x_true_u, x_true_v, x_est_x, x_est_y, x_est_u, x_est_v, z_meas_x, z_meas_y;
	for (size_t i = 0; i < x_true.size() - 1; i++) {
		x_true_x.push_back(x_true.at(i)(0));
		x_true_y.push_back(x_true.at(i)(1));
		x_true_u.push_back(x_true.at(i)(2));
		x_true_v.push_back(x_true.at(i)(3));
		x_est_x.push_back(x_est.at(i).mean()(0));
		x_est_y.push_back(x_est.at(i).mean()(1));
		x_est_u.push_back(x_est.at(i).mean()(2));
		x_est_v.push_back(x_est.at(i).mean()(3));
		z_meas_x.push_back(z_meas.at(i)(0));
		z_meas_y.push_back(z_meas.at(i)(1));
	}
	time.pop_back();

	Gnuplot gp;
	gp << "set terminal qt size 1600,1000\n"; // Modified to make plot larger
	gp << "set multiplot layout 2,1\n";
	gp << "set title 'Position'\n";
	gp << "set xlabel 'x [m]'\n";
	gp << "set ylabel 'y [m]'\n";
	gp << "plot '-' with lines title 'True', '-' with lines title 'Estimate', '-' with points title 'Measurements' ps 2\n"; // Modified to make dots larger
	gp.send1d(std::make_tuple(x_true_x, x_true_y));
	gp.send1d(std::make_tuple(x_est_x, x_est_y));
	gp.send1d(std::make_tuple(z_meas_x, z_meas_y));
	gp << "set title 'Velocity'\n";
	gp << "set xlabel 't [s]'\n";
	gp << "set ylabel 'u,v [m/s]'\n";
	gp << "plot '-' with lines title 'True v', "
	   << "'-' with lines title 'Estimate v', "
	   << "'-' with lines title 'True u', "
	   << "'-' with lines title 'Estimate u'\n";
	gp.send1d(std::make_tuple(time, x_true_v));
	gp.send1d(std::make_tuple(time, x_est_v));
	gp.send1d(std::make_tuple(time, x_true_u));
	gp.send1d(std::make_tuple(time, x_est_u));
	gp << "unset multiplot\n";
}