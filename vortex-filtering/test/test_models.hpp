#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

class SimpleDynamicModel : public vortex::models::interface::DynamicModelCTLTV<2> {
public:
  using BaseI = vortex::models::interface::DynamicModelI<2>;
  using typename BaseI::Mat_xx;
  using typename BaseI::Vec_x;
  constexpr static int N_DIM_x = BaseI::N_DIM_x;

  Mat_xx A_c(const Vec_x & = Vec_x::Zero()) const override { return -Mat_xx::Identity(); }

  Mat_vv Q_c(const Vec_x & = Vec_x::Zero()) const override { return Mat_xx::Identity(); }
};

class NonlinearModel1 : public vortex::models::interface::DynamicModelI<1, 1, 1> {
public:
  using typename DynamicModelI<1, 1, 1>::Vec_x;
  using typename DynamicModelI<1, 1, 1>::Mat_xx;
  using typename DynamicModelI<1, 1, 1>::Mat_xv;
  using typename DynamicModelI<1, 1, 1>::Vec_v;

  NonlinearModel1(double std_dev) : cov_(std_dev * std_dev) {}

  Vec_x f_d(double, const Vec_x &x, const Vec_u & = Vec_u::Zero(), const Vec_v &v = Vec_v::Zero()) const override
  {
    Vec_x x_next;
    x_next << std::sin(x(0)) + v(0);
    return x_next;
  }

  Mat_vv Q_d(double = 0.0, const Vec_x & = Vec_x::Zero()) const override { return Mat_xx::Identity() * cov_; }

private:
  const double cov_;
};

// https://en.wikipedia.org/wiki/Lorenz_system
class LorenzAttractorCT : public vortex::models::interface::DynamicModelCT<3> {
public:
  using BaseI = vortex::models::interface::DynamicModelI<3>;
  using typename BaseI::Vec_u;
  using typename BaseI::Vec_v;
  using typename BaseI::Vec_x;

  using typename BaseI::Mat_vv;
  using typename BaseI::Mat_xx;

  LorenzAttractorCT(double std_dev) : cov_(std_dev * std_dev), sigma_(10.0), beta_(8.0 / 3.0), rho_(28.0) {}

  Vec_x f_c(const Vec_x &x, const Vec_u & = Vec_u::Zero(), const Vec_v &v = Vec_v::Zero()) const override
  {
    Vec_x x_next;
    x_next << sigma_ * (x(1) - x(0)), x(0) * (rho_ - x(2)) - x(1), x(0) * x(1) - beta_ * x(2);
    x_next += v;
    return x_next;
  }

  Mat_vv Q_d(double dt, const Vec_x & = Vec_x::Zero()) const override { return Mat_xx::Identity() * cov_ * dt; }

private:
  const double cov_;
  const double sigma_;
  const double beta_;
  const double rho_;
};

class LorenzAttractorCTLTV : public vortex::models::interface::DynamicModelCTLTV<3> {
public:
  using BaseI = vortex::models::interface::DynamicModelI<3>;
  using typename BaseI::Vec_u;
  using typename BaseI::Vec_v;
  using typename BaseI::Vec_x;

  using typename BaseI::Mat_vv;
  using typename BaseI::Mat_xv;
  using typename BaseI::Mat_xx;

  LorenzAttractorCTLTV(double std_dev) : cov_(std_dev * std_dev), sigma_(10.0), beta_(8.0 / 3.0), rho_(28.0) {}

  Mat_xx A_c(const Vec_x &x) const override
  {
    Mat_xx A;
    // clang-format off
        A << -sigma_  , sigma_, 0.0   ,
             rho_-x(2), -1.0  , -x(0) ,
             x(1)     , x(0)  , -beta_;
    // clang-format on
    return A;
  }

  Mat_vv Q_c(const Vec_x & = Vec_x::Zero()) const override { return Mat_xx::Identity() * cov_; }

private:
  const double cov_;
  const double sigma_;
  const double beta_;
  const double rho_;
};
