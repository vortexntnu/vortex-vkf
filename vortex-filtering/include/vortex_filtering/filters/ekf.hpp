/**
 * @file ekf.hpp
 * @author Eirik Kolås
 * @brief Extended Kalman Filter.
 * @version 0.1
 * @date 2023-10-26
 *
 * @copyright Copyright (c) 2023
 */

#pragma once
#include <tuple>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::filter {

/**
 * @brief Extended Kalman Filter.
 * @tparam DynModT Dynamic model type derived from `vortex::models::interface::DynamicModelLTV`
 * @tparam SensModT Sensor model type derived from `vortex::models::interface::SensorModelLTV`
 */
template <concepts::DynamicModelLTVWithDefinedSizes DynModT, concepts::SensorModelLTVWithDefinedSizes SensModT> class EKF {
public:
  static constexpr int N_DIM_x = DynModT::N_DIM_x;
  static constexpr int N_DIM_z = SensModT::N_DIM_z;
  static constexpr int N_DIM_u = DynModT::N_DIM_u;
  static constexpr int N_DIM_v = DynModT::N_DIM_v;
  static constexpr int N_DIM_w = SensModT::N_DIM_w;

  using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  EKF() = delete;

  /** Perform one EKF prediction step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Previous state estimate
   * @param u T::Vec_x Input. Not used, set to zero.
   * @return std::pair<T::Gauss_x, T::Gauss_z> Predicted state, predicted measurement
   * @throws std::runtime_error if dyn_mod or sens_mod are not of the DynamicModelT or SensorModelT type
   */
  static std::pair<typename T::Gauss_x, typename T::Gauss_z> predict(const DynModT &dyn_mod, const SensModT &sens_mod, double dt, const T::Gauss_x &x_est_prev,
                                                                     const T::Vec_u &u = T::Vec_u::Zero())
  {
    typename T::Gauss_x x_est_pred = dyn_mod.pred_from_est(dt, x_est_prev, u);
    typename T::Gauss_z z_est_pred = sens_mod.pred_from_est(x_est_pred);
    return {x_est_pred, z_est_pred};
  }

  /** Perform one EKF update step
   * @param sens_mod Sensor model
   * @param x_est_pred Predicted state
   * @param z_est_pred Predicted measurement
   * @param z_meas T::Vec_z Measurement
   * @return MultivarGauss Updated state
   * @throws std::runtime_error ifsens_mod is not of the SensorModelT type
   */
  static T::Gauss_x update(const SensModT &sens_mod, const T::Gauss_x &x_est_pred, const T::Gauss_z &z_est_pred, const T::Vec_z &z_meas)
  {
    typename T::Mat_zx C     = sens_mod.C(x_est_pred.mean()); // Measurement matrix
    typename T::Mat_ww R     = sens_mod.R(x_est_pred.mean()); // Measurement noise covariance
    typename T::Mat_zw H     = sens_mod.H(x_est_pred.mean()); // Measurement noise cross-covariance
    typename T::Mat_xx P     = x_est_pred.cov();              // State covariance
    typename T::Mat_zz S_inv = z_est_pred.cov_inv();          // Inverse of the predicted measurement covariance
    typename T::Mat_xx I     = T::Mat_xx::Identity(N_DIM_x, N_DIM_x);

    typename T::Mat_xz W         = P * C.transpose() * S_inv; // Kalman gain
    typename T::Vec_z innovation = z_meas - z_est_pred.mean();

    typename T::Vec_x state_upd_mean = x_est_pred.mean() + W * innovation;
    // Use the Joseph form of the covariance update to ensure positive definiteness
    typename T::Mat_xx state_upd_cov = (I - W * C) * P * (I - W * C).transpose() + W * H * R * H.transpose() * W.transpose();

    return {state_upd_mean, state_upd_cov};
  }

  /** Perform one EKF prediction and update step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Previous state estimate
   * @param z_meas T::Vec_z Measurement
   * @param u T::Vec_x Input
   * @return Updated state, predicted state, predicted measurement
   */
  static std::tuple<typename T::Gauss_x, typename T::Gauss_x, typename T::Gauss_z>
  step(const DynModT &dyn_mod, const SensModT &sens_mod, double dt, const T::Gauss_x &x_est_prev, const T::Vec_z &z_meas, const T::Vec_u &u = T::Vec_u::Zero())
  {
    auto [x_est_pred, z_est_pred] = predict(dyn_mod, sens_mod, dt, x_est_prev, u);

    typename T::Gauss_x x_est_upd = update(sens_mod, x_est_pred, z_est_pred, z_meas);
    return {x_est_upd, x_est_pred, z_est_pred};
  }

  [[deprecated("use const DynModT& and const SensModT&")]] static std::pair<typename T::Gauss_x, typename T::Gauss_z>
  predict(std::shared_ptr<DynModT> dyn_mod, std::shared_ptr<SensModT> sens_mod, double dt, const T::Gauss_x &x_est_prev, const T::Vec_u &u = T::Vec_u::Zero())
  {
    return predict(*dyn_mod, *sens_mod, dt, x_est_prev, u);
  }

  [[deprecated("use const SensModT&")]] static T::Gauss_x update(std::shared_ptr<SensModT> sens_mod, const T::Gauss_x &x_est_pred, const T::Gauss_z &z_est_pred,
                                                                 const T::Vec_z &z_meas)
  {
    return update(*sens_mod, x_est_pred, z_est_pred, z_meas);
  }

  [[deprecated("use const DynModT& and const SensModT&")]] static std::tuple<typename T::Gauss_x, typename T::Gauss_x, typename T::Gauss_z>
  step(std::shared_ptr<DynModT> dyn_mod, std::shared_ptr<SensModT> sens_mod, double dt, const T::Gauss_x &x_est_prev, const T::Vec_z &z_meas,
       const T::Vec_u &u = T::Vec_u::Zero())
  {
    return step(*dyn_mod, *sens_mod, dt, x_est_prev, z_meas, u);
  }
};

} // namespace vortex::filter