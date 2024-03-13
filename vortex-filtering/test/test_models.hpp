#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/numerical_integration/erk_methods.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

using vortex::models::interface::DynamicModel;
using vortex::models::interface::DynamicModelCTLTV;
class SimpleDynamicModel : public DynamicModelCTLTV<2> {
public:
  constexpr static int N_DIM_x = DynamicModel<2>::N_DIM_x;
  constexpr static int N_DIM_u = DynamicModel<2>::N_DIM_u;
  constexpr static int N_DIM_v = DynamicModel<2>::N_DIM_v;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  T::Mat_xx A_c(const T::Vec_x & = T::Vec_x::Zero()) const override { return -T::Mat_xx::Identity(); }

  T::Mat_vv Q_c(const T::Vec_x & = T::Vec_x::Zero()) const override { return T::Mat_xx::Identity(); }
};

class NonlinearModel1 : public DynamicModel<1, 1, 1> {
public:
  constexpr static int N_DIM_x = DynamicModel<1, 1, 1>::N_DIM_x;
  constexpr static int N_DIM_u = DynamicModel<1, 1, 1>::N_DIM_u;
  constexpr static int N_DIM_v = DynamicModel<1, 1, 1>::N_DIM_v;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  NonlinearModel1(double std_dev)
      : cov_(std_dev * std_dev)
  {
  }

  T::Vec_x f_d(double, const T::Vec_x &x, const T::Vec_u & = T::Vec_u::Zero(), const T::Vec_v &v = T::Vec_v::Zero()) const override
  {
    typename T::Vec_x x_next;
    x_next << std::sin(x(0)) + v(0);
    return x_next;
  }

  T::Mat_vv Q_d(double = 0.0, const T::Vec_x & = T::Vec_x::Zero()) const override { return T::Mat_xx::Identity() * cov_; }

private:
  const double cov_;
};

// https://en.wikipedia.org/wiki/Lorenz_system
class LorenzAttractorCT : public DynamicModel<3> {
public:
  static constexpr int N_DIM_x = DynamicModel<3>::N_DIM_x;
  static constexpr int N_DIM_u = DynamicModel<3>::N_DIM_u;
  static constexpr int N_DIM_v = DynamicModel<3>::N_DIM_v;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  LorenzAttractorCT(double std_dev)
      : cov_(std_dev * std_dev)
      , sigma_(10.0)
      , beta_(8.0 / 3.0)
      , rho_(28.0)
  {
  }

  T::Vec_x f_c(const T::Vec_x &x, const T::Vec_u & = T::Vec_u::Zero(), const T::Vec_v &v = T::Vec_v::Zero()) const
  {
    typename T::Vec_x x_next;
    x_next << sigma_ * (x(1) - x(0)), x(0) * (rho_ - x(2)) - x(1), x(0) * x(1) - beta_ * x(2);
    x_next += v;
    return x_next;
  }

  T::Vec_x f_d(double dt, const T::Vec_x &x, const T::Vec_u &u = T::Vec_u::Zero(), const T::Vec_v &v = T::Vec_v::Zero()) const override
  {
    using Dyn_mod_func = std::function<T::Vec_x(double t, const T::Vec_x &x)>;

    Dyn_mod_func f_c = [this, &u, &v](double, const T::Vec_x &x) { return this->f_c(x, u, v); };
    return vortex::integrator::RK4<N_DIM_x>::integrate(f_c, dt, x);
  }

  T::Mat_vv Q_d(double dt, const T::Vec_x & = T::Vec_x::Zero()) const override { return T::Mat_xx::Identity() * cov_ * dt; }

private:
  const double cov_;
  const double sigma_;
  const double beta_;
  const double rho_;
};

class LorenzAttractorCTLTV : public vortex::models::interface::DynamicModelCTLTV<3> {
public:
  static constexpr int N_DIM_x = DynamicModelCTLTV<3>::N_DIM_x;
  static constexpr int N_DIM_u = DynamicModelCTLTV<3>::N_DIM_u;
  static constexpr int N_DIM_v = DynamicModelCTLTV<3>::N_DIM_v;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  LorenzAttractorCTLTV(double std_dev) : cov_(std_dev * std_dev), sigma_(10.0), beta_(8.0 / 3.0), rho_(28.0) {}

  T::Mat_xx A_c(const T::Vec_x &x) const override
  {
    typename T::Mat_xx A;
    // clang-format off
        A << -sigma_  , sigma_, 0.0   ,
             rho_-x(2), -1.0  , -x(0) ,
             x(1)     , x(0)  , -beta_;
    // clang-format on
    return A;
  }

  T::Mat_vv Q_c(const T::Vec_x & = T::Vec_x::Zero()) const override { return T::Mat_xx::Identity() * cov_; }

private:
  const double cov_;
  const double sigma_;
  const double beta_;
  const double rho_;
};
