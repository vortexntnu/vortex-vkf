#include <cmath>
#include <functional>
#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <vortex_filtering/numerical_integration/erk_methods.hpp>
#include <vortex_filtering/utils/plotting.hpp>

namespace sin_func_test {
constexpr int N_DIM_x = 1, N_DIM_z = 1, N_DIM_u = 1, N_DIM_v = 1, N_DIM_w = 1;
using Vec_x        = Eigen::Vector<double, N_DIM_x>;
using Vec_z        = Eigen::Vector<double, N_DIM_z>;
using Vec_u        = Eigen::Vector<double, N_DIM_u>;
using Vec_v        = Eigen::Vector<double, N_DIM_v>;
using Vec_w        = Eigen::Vector<double, N_DIM_w>;
using Dyn_mod_func = std::function<Vec_x(double t0, const Vec_x &x0)>;

// Make functions to test the RK methods
Vec_x sin_func(double t, const Vec_x &x)
{
  Vec_x x_dot;
  x_dot << std::pow(std::sin(t), 2) * x(0);
  return x_dot;
}
Vec_x sin_func_exact(double t, const Vec_x &x0)
{
  Vec_x x_kp1;
  x_kp1 << x0 * exp(0.5 * (t - sin(t) * cos(t)));
  return x_kp1;
}

class NumericalIntegration : public ::testing::Test {
protected:
  NumericalIntegration()
  {
    dt = 1e-3;
    init(0, Vec_x::Zero());
    u.setZero();
    v.setZero();
  }
  double dt;

  Vec_u u;
  Vec_v v;

  std::vector<double> t;
  std::vector<Eigen::VectorXd> x_est;
  std::vector<Eigen::VectorXd> x_exact;

  void init(double t0, Vec_x x0)
  {
    t.clear();
    x_est.clear();
    x_exact.clear();
    t.push_back(t0);
    x_est.push_back(x0);
    x_exact.push_back(x0);
  }

  template <class rk_method> void run_iterations(Dyn_mod_func f, Dyn_mod_func f_exact, size_t num_iters, double tolerance)
  {
    for (size_t i = 0; i < num_iters; i++) {
      x_est.push_back(rk_method::integrate(f, dt, x_est.at(i), t.at(i)));
      t.push_back(t.at(i) + dt);
      x_exact.push_back(f_exact(t.at(i + 1), x_exact.at(0)));
    }
    std::cout << "True: " << x_exact.back()(0) << std::endl;
    std::cout << "Approx: " << x_est.back()(0) << std::endl;
    std::cout << "Error: " << x_est.back()(0) - x_exact.back()(0) << std::endl;
    EXPECT_NEAR(x_est.back()(0), x_exact.back()(0), tolerance);
  }

  void plot_result(std::string title = "ERK convergence")
  {
    #ifdef GNUPLOT_ENABLE
    // Plot first state and true solution against time
    Gnuplot gp;
    gp << "set terminal wxt size 1200,800\n";
    gp << "set title '" << title << "'\n";
    gp << "set xlabel 'Time [s]'\n";
    gp << "set ylabel 'x'\n";
    gp << "set grid\n";
    gp << "set key left top\n";
    gp << "plot '-' with lines title 'True', '-' with lines title 'Approx'\n";
    gp.send1d(std::make_tuple(t, vortex::plotting::extract_state_series(x_exact, 0)));
    gp.send1d(std::make_tuple(t, vortex::plotting::extract_state_series(x_est, 0)));
    #endif
    (void)title;
  }
};

TEST_F(NumericalIntegration, RK4sinFunc)
{
  Vec_x x0;
  x0 << 1;
  init(0, x0);
  size_t n_iters = 5000;

  // Expected error is O(dt^4)
  run_iterations<vortex::integrator::RK4<N_DIM_x>>(sin_func, sin_func_exact, n_iters, 1e-4);
  // plot_result();
}

TEST_F(NumericalIntegration, EulerSinFunc)
{
  Vec_x x0;
  x0 << 1;
  init(0, x0);
  size_t n_iters = 5000;

  // Expected error is O(dt)
  run_iterations<vortex::integrator::Forward_Euler<N_DIM_x>>(sin_func, sin_func_exact, n_iters, 1e-1);
  plot_result("Euler Sin Func");
}

TEST_F(NumericalIntegration, ButcherMidpointSinFunc)
{
  Vec_x x0;
  x0 << 1;
  init(0, x0);
  constexpr int n_stages = 2;
  Eigen::Matrix<double, n_stages, n_stages> A;
  Eigen::Vector<double, n_stages> b;
  Eigen::Vector<double, n_stages> c;

  // Midpoint method
  A << 0, 0, 1 / 2.0, 0;

  b << 0, 1;
  c << 0, 1 / 2.0;

  auto midpoint  = std::make_shared<vortex::integrator::Butcher<N_DIM_x, n_stages>>(A, b, c);
  size_t n_iters = 5000;

  // Expected error is O(dt^2)
  Vec_x exact      = sin_func_exact(dt * n_iters, x0);
  double tolerance = 1e-5;
  size_t num_iters = 5000;
  for (size_t i = 0; i < num_iters; i++) {
    x_est.push_back(midpoint->integrate(sin_func, dt, x_est.at(i), t.at(i)));
    t.push_back(t.at(i) + dt);
    x_exact.push_back(sin_func_exact(t.at(i + 1), x_exact.at(0)));
  }
  std::cout << "True: " << exact << std::endl;
  std::cout << "Approx: " << x_est.back()(0) << std::endl;
  std::cout << "Error: " << x_est.back()(0) - x_exact.back()(0) << std::endl;
  EXPECT_NEAR(x_est.back()(0), x_exact.back()(0), tolerance);
}

TEST_F(NumericalIntegration, ButcherRKDPsinFunc)
{
  Vec_x x0;
  x0 << 1;
  init(0, x0);
  constexpr int n_stages = 7;
  Eigen::Matrix<double, n_stages, n_stages> A;
  Eigen::Vector<double, n_stages> b;
  Eigen::Vector<double, n_stages> c;

  // Dormand Prince (RKDP) method
  // clang-format off
	A << 0           ,  0           , 0           ,  0        ,  0             , 0         , 0,
		 1/5.0       ,  0           , 0           ,  0        ,  0             , 0         , 0,
		 3/40.0      ,  9/40.0      , 0           ,  0        ,  0             , 0         , 0,
		 44/45.0     , -56/15.0     , 32/9.0      ,  0        ,  0             , 0         , 0,
		 19372/6561.0, -25360/2187.0, 64448/6561.0, -212/729.0,  0             , 0         , 0,
		 9017/3168.0 , -355/33.0    , 46732/5247.0,  49/176.0 , -5103/18656.0  , 0         , 0,
		 35/384.0    ,  0           , 500/1113.0  ,  125/192.0, -2187/6784.0   , 11/84.0   , 0;

	b << 35/384.0    ,  0           , 500/1113.0  ,  125/192.0, -2187/6784.0   , 11/84.0   , 0,	// Error of order O(dt^5)

	c << 0           ,  1/5.0       , 3/10.0      ,  4/5.0    ,  8/9.0         , 1         , 1;
  // clang-format on
  auto rkdp      = std::make_shared<vortex::integrator::Butcher<N_DIM_x, n_stages>>(A, b, c);
  size_t n_iters = 5000;

  // Expected error is O(dt^5)
  Vec_x exact      = sin_func_exact(dt * n_iters, x0);
  double tolerance = 1e-8;

  for (size_t i = 0; i < n_iters; i++) {
    x_est.push_back(rkdp->integrate(sin_func, dt, x_est.at(i), t.at(i)));
    t.push_back(t.at(i) + dt);
    x_exact.push_back(sin_func_exact(t.at(i + 1), x_exact.at(0)));
  }
  std::cout << "True: " << exact << std::endl;
  std::cout << "Approx: " << x_est.back()(0) << std::endl;
  std::cout << "Error: " << x_est.back()(0) - x_exact.back()(0) << std::endl;
  EXPECT_NEAR(x_est.back()(0), x_exact.back()(0), tolerance);
}

TEST_F(NumericalIntegration, ODE45sinFuncRealTime)
{
  Vec_x x0;
  x0 << 1;
  init(0, x0);
  auto rk45      = std::make_shared<vortex::integrator::ODE45<N_DIM_x>>(1e-20, 1e-20);
  size_t n_iters = 5000;

  // Expected error is O(dt^5)
  Vec_x exact      = sin_func_exact(dt * n_iters, x0);
  double tolerance = 1e-8;
  for (size_t i = 0; i < n_iters; i++) {
    x_est.push_back(rk45->integrate(sin_func, dt, x_est.at(i), t.at(i)));
    t.push_back(t.at(i) + dt);
    x_exact.push_back(sin_func_exact(t.at(i + 1), x_exact.at(0)));
  }
  std::cout << "True: " << exact << std::endl;
  std::cout << "Approx: " << x_est.back()(0) << std::endl;
  std::cout << "Error: " << x_est.back()(0) - x_exact.back()(0) << std::endl;
  EXPECT_NEAR(x_est.back()(0), x_exact.back()(0), tolerance);
}

TEST_F(NumericalIntegration, ODE45sinFuncLongStep)
{
  Vec_x x0;
  x0 << 1;
  init(0, x0);
  auto rk45      = std::make_shared<vortex::integrator::ODE45<N_DIM_x>>(1e-20, 1e-20, 100000);
  dt             = 5;
  size_t n_iters = 1;

  // Expected error is O(dt^5)
  Vec_x exact      = sin_func_exact(dt * n_iters, x0);
  double tolerance = 1e-6;

  for (size_t i = 0; i < n_iters; i++) {
    x_est.push_back(rk45->integrate(sin_func, dt, x_est.at(i), t.at(i)));
    t.push_back(t.at(i) + dt);
    x_exact.push_back(sin_func_exact(t.at(i + 1), x_exact.at(0)));
  }
  std::cout << "True: " << exact << std::endl;
  std::cout << "Approx: " << x_est.back()(0) << std::endl;
  std::cout << "Error: " << x_est.back()(0) - x_exact.back()(0) << std::endl;
  EXPECT_NEAR(x_est.back()(0), x_exact.back()(0), tolerance);
}

} // namespace sin_func_test

namespace van_der_pol_test {

constexpr int N_DIM_x = 2, N_DIM_z = 1, N_DIM_u = 1, N_DIM_v = 2, N_DIM_w = 1;
using Vec_x        = Eigen::Vector<double, N_DIM_x>;
using Vec_z        = Eigen::Vector<double, N_DIM_z>;
using Vec_u        = Eigen::Vector<double, N_DIM_u>;
using Vec_v        = Eigen::Vector<double, N_DIM_v>;
using Vec_w        = Eigen::Vector<double, N_DIM_w>;
using Dyn_mod_func = std::function<Vec_x(double t0, const Vec_x &x0)>;

// van der Pol oscillator
Vec_x vdp(double t, const Vec_x &x, const Vec_u &u, const Vec_v &v)
{
  (void)t;
  (void)v;
  Vec_x x_dot;
  x_dot << x(1), (1 - std::pow(x(0), 2)) * x(1) - x(0) + u(0);
  return x_dot;
}

} // namespace van_der_pol_test
