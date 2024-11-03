
#pragma once
#include <manif/manif.h>
#include <memory>
#include <tuple>
#include <vortex_filtering/models/interfaces/dynamic_model_ltv.hpp>
#include <vortex_filtering/models/interfaces/sensor_model_ltv.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::filter {

template <typename Type_x, typename Type_z, typename Type_u, typename Type_v, typename Type_w> class LIEKF {
public:
  using Mx = Type_x; // Lie group of the state
  using Mz = Type_z; // Lie group of the measurement
  using Mu = Type_u; // Lie group of the input
  using Mv = Type_v; // Lie group of the process noise
  using Mw = Type_w; // Lie group of the measurement noise

  static constexpr int N_DIM_x = Mx::DoF;
  static constexpr int N_DIM_z = Mz::DoF;
  static constexpr int N_DIM_u = Mu::DoF;
  static constexpr int N_DIM_v = Mv::DoF;
  static constexpr int N_DIM_w = Mw::DoF;

  using T = vortex::Types_xzuvw<Mx::DoF, Mz::DoF, Mu::DoF, Mv::DoF, Mw::DoF>;

  using Tx = typename Mx::Tangent; // Tangent space of the Lie group of the state
  using Tz = typename Mz::Tangent; // Tangent space of the Lie group of the measurement
  using Tu = typename Mu::Tangent; // Tangent space of the Lie group of the input
  using Tv = typename Mv::Tangent; // Tangent space of the Lie group of the process noise
  using Tw = typename Mw::Tangent; // Tangent space of the Lie group of the measurement noise

  using DynMod  = vortex::model::interface::DynamicModelLTV<Mx, Mu, Mv>;
  using SensMod = vortex::model::interface::SensorModelLTV<Mx, Mz, Mw>;

  using Gauss_x = vortex::prob::LieGroupGauss<Mx>;
  using Gauss_z = vortex::prob::LieGroupGauss<Mz>;

  LIEKF() = delete;

  /** Perform one EKF prediction step
   * @param dyn_mod Dynamic model
   * @param dt Time step
   * @param x_est_prev Gauss_x Previous state estimate
   * @param u Vec_x Input. Not used, set to zero.
   * @return std::pair<Gauss_x, Gauss_z> Predicted state, predicted measurement
   * @throws std::runtime_error if dyn_mod is not of the DynamicModelT type
   */
  static std::pair<Gauss_x, Gauss_z> predict(std::unique_ptr<const DynMod> dyn_mod,
                                             std::unique_ptr<const SensMod> sens_mod, double dt,
                                             const Gauss_x &x_est_prev, const Mu &u = Mu::Identity())
  {
    typename T::Mat_xx P = x_est_prev.cov();
    typename T::Mat_xx A = dyn_mod.J_f_x(dt, x_est_prev.mean());
    typename T::Mat_zx H = sens_mod.J_g_x(x_est_prev.mean());

    typename T::Mat_vv Q = dyn_mod.Q_x(dt, x_est_prev.mean());
    typename T::Mat_ww R = sens_mod.R_z(x_est_prev.mean());

    auto Iv = Mv::Identity();
    auto Iw = Mw::Identity();

    auto x_next = dyn_mod->f(dt, x_est_prev.mean(), u, Iv);
    auto P_next = A * P * A.transpose() + Q;

    auto z_next = sens_mod->g(x_next, Iw);
    auto S_next = H * P_next * H.transpose() + R;

    Gauss_x x_est_pred = {x_next, P_next};
    Gauss_z z_est_pred = {z_next, S_next};

    return {x_est_pred, z_est_pred};
  }

  /** Perform one EKF update step
   * @param sens_mod Sensor model
   * @param x_est_pred Gauss_x Predicted state
   * @param z_est_pred Gauss_z Predicted measurement
   * @param z_meas Vec_z Measurement
   * @return MultivarGauss Updated state
   */
  static Gauss_x update(const SensMod &sens_mod, const Gauss_x &x_est_pred, const Gauss_z &z_est_pred, const Mz &z_meas)
  {
    auto X_hat = x_est_pred.mean();
    auto Y     = z_meas;

    typename T::Mat_zx H     = sens_mod.J_g_x(X_hat);     // Measurement matrix
    typename T::Mat_ww R     = sens_mod.R_z(X_hat);       // Measurement noise covariance in noise space
    typename T::Mat_xx P     = x_est_pred.cov();          // Predicted state covariance
    typename T::Mat_zz S_inv = z_est_pred.cov_inv();      // Inverse of the predicted measurement covariance
    typename T::Mat_xx I     = T::Mat_xx::Identity();     // Identity matrix
    typename T::Mat_xz K     = P * H.transpose() * S_inv; // Kalman gain

    auto b = sens_mod->b();

    Tx innovation     = X_hat.inv().act(Y) - b;
    Mx state_upd_mean = X_hat + K * innovation;

    typename T::Mat_xx state_upd_cov = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

    return {state_upd_mean, state_upd_cov};
  }

  /** Perform one EKF prediction and update step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Gauss_x Previous state estimate
   * @param z_meas Vec_z Measurement
   * @param u Vec_x Input
   * @return std::pair<Gauss_x, Gauss_z> Updated state, updated measurement
   */
  static std::tuple<Gauss_x, Gauss_x, Gauss_z> step(const DynMod &dyn_mod, const SensMod &sens_mod, double dt,
                                                    const Gauss_x &x_est_prev, const Mz &z_meas,
                                                    const Tu &u = Tu::Zero())
  {
    auto [x_est_pred, z_est_pred] = predict(dyn_mod, sens_mod, dt, x_est_prev, u);
    auto x_est_upd                = update(sens_mod, x_est_pred, z_est_pred, z_meas);
    return {x_est_upd, x_est_pred, z_est_pred};
  }
};

} // namespace vortex::filter