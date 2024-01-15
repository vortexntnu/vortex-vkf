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

namespace vortex::filter {

/**
 * @brief Extended Kalman Filter.
 * @tparam n_dim_x State dimension.
 * @tparam n_dim_z Measurement dimension.
 * @tparam n_dim_u Input dimension (Default: n_dim_x)
 * @tparam n_dim_v Process noise dimension (Default: n_dim_x)
 * @tparam n_dim_w Measurement noise dimension (Default: n_dim_z)
 * @note The EKF only works with models derived from `vortex::models::DynamicModelLTV` and `vortex::models::SensorModelLTV`.
 */
template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u = n_dim_x, size_t n_dim_v = n_dim_x, size_t n_dim_w = n_dim_z>
class EKF {
public:
  static constexpr int N_DIM_x = (int)n_dim_x;
  static constexpr int N_DIM_z = (int)n_dim_z;
  static constexpr int N_DIM_u = (int)n_dim_u;
  static constexpr int N_DIM_v = (int)n_dim_v;
  static constexpr int N_DIM_w = (int)n_dim_w;

  using Vec_x = Eigen::Vector<double, N_DIM_x>;
  using Vec_z = Eigen::Vector<double, N_DIM_z>;
  using Vec_u = Eigen::Vector<double, N_DIM_u>;
  using Vec_v = Eigen::Vector<double, N_DIM_v>;
  using Vec_w = Eigen::Vector<double, N_DIM_w>;

  using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
  using Mat_xz = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
  using Mat_xv = Eigen::Matrix<double, N_DIM_x, N_DIM_v>;
  using Mat_xw = Eigen::Matrix<double, N_DIM_x, N_DIM_w>;

  using Mat_zx = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
  using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
  using Mat_zw = Eigen::Matrix<double, N_DIM_z, N_DIM_w>;

  using Mat_vx = Eigen::Matrix<double, N_DIM_v, N_DIM_x>;
  using Mat_vv = Eigen::Matrix<double, N_DIM_v, N_DIM_v>;
  using Mat_vw = Eigen::Matrix<double, N_DIM_v, N_DIM_w>;

  using Mat_wx = Eigen::Matrix<double, N_DIM_w, N_DIM_x>;
  using Mat_wv = Eigen::Matrix<double, N_DIM_w, N_DIM_v>;
  using Mat_ww = Eigen::Matrix<double, N_DIM_w, N_DIM_w>;

  using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
  using Gauss_z = prob::MultiVarGauss<N_DIM_z>;
  using Gauss_v = prob::MultiVarGauss<N_DIM_v>;
  using Gauss_w = prob::MultiVarGauss<N_DIM_w>;

  using DynModI     = models::interface::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v>;
  using SensModI    = models::interface::SensorModelLTV<N_DIM_x, N_DIM_z, N_DIM_w>;

  using DynModIPtr  = typename DynModI::SharedPtr;
  using SensModIPtr = typename SensModI::SharedPtr;

  /** Construct a new EKF object
   */
  EKF() = default;

  /** Perform one EKF prediction step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Previous state estimate
   * @param u Vec_x Input. Not used, set to zero.
   * @return std::pair<Gauss_x, Gauss_z> Predicted state, predicted measurement
   * @throws std::runtime_error if dyn_mod or sens_mod are not of the DynamicModelT or SensorModelT type
   */
  static std::pair<Gauss_x, Gauss_z> predict(const DynModIPtr &dyn_mod, const SensModIPtr &sens_mod, double dt, const Gauss_x &x_est_prev,
                                      const Vec_u &u = Vec_u::Zero())
  {
    Gauss_x x_est_pred = dyn_mod->pred_from_est(dt, x_est_prev, u);
    Gauss_z z_est_pred = sens_mod->pred_from_est(x_est_pred);
    return {x_est_pred, z_est_pred};
  }

  /** Perform one EKF update step
   * @param sens_mod Sensor model
   * @param x_est_pred Predicted state
   * @param z_est_pred Predicted measurement
   * @param z_meas Vec_z Measurement
   * @return MultivarGauss Updated state
   * @throws std::runtime_error ifsens_mod is not of the SensorModelT type
   */
  static Gauss_x update(const SensModIPtr &sens_mod, const Gauss_x &x_est_pred, const Gauss_z &z_est_pred, const Vec_z &z_meas)
  {
    Mat_zx C        = sens_mod->C(x_est_pred.mean());
    Mat_ww R        = sens_mod->R(x_est_pred.mean());
    Mat_zw H        = sens_mod->H(x_est_pred.mean());
    Mat_xx P        = x_est_pred.cov();
    Mat_zz S_inv    = z_est_pred.cov_inv();
    Mat_xx I        = Mat_xx::Identity(N_DIM_x, N_DIM_x);

    Mat_xz W         = P * C.transpose() * S_inv; // Kalman gain
    Vec_z innovation = z_meas - z_est_pred.mean();

    Vec_x state_upd_mean = x_est_pred.mean() + W * innovation;
    // Use the Joseph form of the covariance update to ensure positive definiteness
    Mat_xx state_upd_cov = (I - W * C) * P * (I - W * C).transpose() + W * H * R * H.transpose() * W.transpose();

    return {state_upd_mean, state_upd_cov};
  }

  /** Perform one EKF prediction and update step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Previous state estimate
   * @param z_meas Vec_z Measurement
   * @param u Vec_x Input
   * @return Updated state, predicted state, predicted measurement
   */
  static std::tuple<Gauss_x, Gauss_x, Gauss_z> step(const DynModIPtr &dyn_mod, const SensModIPtr &sens_mod, double dt, const Gauss_x &x_est_prev, const Vec_z &z_meas,
                                             const Vec_u &u = Vec_u::Zero())
  {
    auto [x_est_pred, z_est_pred] = predict(dyn_mod, sens_mod, dt, x_est_prev, u);

    Gauss_x x_est_upd  = update(sens_mod, x_est_pred, z_est_pred, z_meas);
    return {x_est_upd, x_est_pred, z_est_pred};
  }
};

/** @brief EKF with dimensions defined by the dynamic model and sensor model.
 * @tparam DynModT Dynamic model type.
 * @tparam SensModT Sensor model type.
 */
template <typename DynModT, typename SensModT>
using EKF_M = EKF<DynModT::DynModI::N_DIM_x, 
                  SensModT::SensModI::N_DIM_z, 
                  DynModT::DynModI::N_DIM_u, 
                  DynModT::DynModI::N_DIM_v, 
                  SensModT::SensModI::N_DIM_w>;

} // namespace vortex::filter