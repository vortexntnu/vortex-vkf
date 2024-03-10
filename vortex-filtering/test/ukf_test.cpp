
#include <cmath>
#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>

#include "gtest_assertions.hpp"
#include "test_models.hpp"
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

class UKFtest : public ::testing::Test {
protected:
  using Vec_x   = typename NonlinearModel1::Vec_x;
  using Mat_xx  = typename NonlinearModel1::Mat_xx;
  using Gauss_x = typename NonlinearModel1::Gauss_x;

  using IdentitySensorModel = vortex::models::IdentitySensorModel<1, 1>;

  using Vec_z   = typename IdentitySensorModel::Vec_z;
  using Mat_zz  = typename IdentitySensorModel::Mat_zz;
  using Gauss_z = typename IdentitySensorModel::Gauss_z;

  using UKF = vortex::filter::UKF<NonlinearModel1, IdentitySensorModel>;

  static constexpr double Q = 0.1;
  static constexpr double R = 0.1;
  UKFtest() : dynamic_model_(Q), sensor_model_(R) {}


  NonlinearModel1 dynamic_model_;
  IdentitySensorModel sensor_model_;
};

TEST_F(UKFtest, Predict)
{
  // Initial state
  Gauss_x x(Vec_x::Zero(), Mat_xx::Identity());
  double dt = 0.1;
  // Predict
  auto [x_est_pred, z_est_pred] = UKF::predict(dynamic_model_, sensor_model_, dt, x);

  Vec_x x_true = dynamic_model_.f_d(dt, x.mean());
  Vec_z z_true = x_true.head(1);
  ASSERT_TRUE(isApproxEqual(x_est_pred.mean(), x_true, 1e-6));
  ASSERT_TRUE(isApproxEqual(z_est_pred.mean(), z_true, 1e-6));
}

TEST_F(UKFtest, Convergence)
{
  // Random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d_disturbance{0, 1e-3};
  std::normal_distribution<> d_noise{0, 1e-2};

  // Initial state
  Gauss_x x0(4 * Vec_x::Ones(), Mat_xx::Identity());
  // Time step
  double dt = 0.1;
  // Number of steps
  int n_steps = 1000;

  std::vector<double> time;
  std::vector<Vec_x> x_true;
  std::vector<Gauss_x> x_est;
  std::vector<Vec_z> z_meas;
  std::vector<Gauss_z> z_est;

  time.push_back(0);
  x_est.push_back(x0);
  z_meas.push_back(sensor_model_.h(x0.mean()));
  x_true.push_back(x0.mean());
  z_est.push_back({sensor_model_.h(x0.mean()), sensor_model_.R()});
  for (int i = 0; i < n_steps - 1; i++) {
    // Simulate
    Vec_x v;
    v << d_disturbance(gen);
    Vec_z w = Vec_z::Zero();
    w << d_noise(gen);
    Vec_x x_true_i = dynamic_model_.f_d(dt, x_true.back(), Vec_x::Zero(), v);
    Vec_z z_meas_i = sensor_model_.h(x_true_i) + w;
    x_true.push_back(x_true_i);
    z_meas.push_back(z_meas_i);

    // Predict and update
    auto [x_est_upd, x_est_pred, z_est_pred] = UKF::step(dynamic_model_, sensor_model_, dt, x_est.back(), z_meas_i, Vec_x::Zero());

    // Save results
    time.push_back(time.back() + dt);
    x_est.push_back(x_est_upd);
    z_est.push_back(z_est_pred);
  }

  // Check convergence
  double tol = 1e-1;

  // Plot results

  std::vector<double> x_est_mean, z_est_mean, x_est_std, x_p_std, x_m_std;
  for (int i = 0; i < n_steps; i++) {
    x_est_mean.push_back(x_est.at(i).mean()(0));
    z_est_mean.push_back(z_est.at(i).mean()(0));
    x_est_std.push_back(std::sqrt(x_est.at(i).cov()(0, 0)));
    x_p_std.push_back(x_est.at(i).mean()(0) + std::sqrt(x_est.at(i).cov()(0, 0))); // x_est + std
    x_m_std.push_back(x_est.at(i).mean()(0) - std::sqrt(x_est.at(i).cov()(0, 0))); // x_est - std
  }

  #if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set terminal wxt size 1200,800\n";
  gp << "set title 'UKF convergence'\n";
  gp << "set xlabel 'Time [s]'\n";
  gp << "set ylabel 'x'\n";
  gp << "set grid\n";
  gp << "set key left top\n";
  gp << "plot '-' using 1:($2-$3):($2+$3) with filledcurves title 'x_{est} +/- std' fs solid 0.5, "
     << "'-' with lines title 'x_{true}', "
     << "'-' with lines title 'x_{est}', "
     << "'-' with lines title 'z_{meas}' \n";
  gp.send1d(std::make_tuple(time, x_est_mean, x_est_std));
  gp.send1d(std::make_tuple(time, x_true));
  gp.send1d(std::make_tuple(time, x_est_mean));
  gp.send1d(std::make_tuple(time, z_meas));
  #endif

  // Check convergence

  ASSERT_NEAR(x_est.back().mean()(0), x_true.back()(0), tol);
  ASSERT_NEAR(z_est.back().mean()(0), z_meas.back()(0), tol);
}