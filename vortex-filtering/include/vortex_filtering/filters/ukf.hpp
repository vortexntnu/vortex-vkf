/**
 * @file ukf.hpp
 * @author Eirik Kolås
 * @brief UKF. Does not implement the square root version of the UKF.
 * Does not implement a stand-alone update step.
 * @version 0.1
 * @date 2023-11-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once
#include <tuple>
#include <vector>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/types/type_aliases.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace filter {

/** Unscented Kalman Filter
 * @tparam DynModT Dynamic model type derived from `vortex::models::interface::DynamicModel`
 * @tparam SensModT Sensor model type derived from `vortex::models::interface::SensorModel`
 * @tparam alpha Parameter for spread of sigma points (default 1.0)
 * @tparam beta Parameter for weighting of mean in covariance calculation (default 2.0)
 * @tparam kappa Parameter for adding additional spread to sigma points (default 0.0)
 */
template <concepts::DynamicModelWithDefinedSizes DynModT, concepts::SensorModelWithDefinedSizes SensModT, double alpha = 1.0, double beta = 2.0,
          double kappa = 0.0>
class UKF {
public:
  static constexpr int N_DIM_x = DynModT::N_DIM_x;
  static constexpr int N_DIM_u = DynModT::N_DIM_u;
  static constexpr int N_DIM_z = SensModT::N_DIM_z;
  static constexpr int N_DIM_v = DynModT::N_DIM_v;
  static constexpr int N_DIM_w = SensModT::N_DIM_w;

  using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  static constexpr int N_DIM_a           = N_DIM_x + N_DIM_v + N_DIM_w; // Augmented state dimension
  static constexpr size_t N_SIGMA_POINTS = 2 * N_DIM_a + 1;             // Number of sigma points

  using Vec_a     = Eigen::Vector<double, N_DIM_a>;                 // Augmented state vector
  using Mat_aa    = Eigen::Matrix<double, N_DIM_a, N_DIM_a>;        // Augmented state covariance matrix
  using Mat_x2ap1 = Eigen::Matrix<double, N_DIM_x, N_SIGMA_POINTS>; // Matrix for sigma points of x
  using Mat_z2ap1 = Eigen::Matrix<double, N_DIM_z, N_SIGMA_POINTS>; // Matrix for sigma points of z
  using Mat_a2ap1 = Eigen::Matrix<double, N_DIM_a, N_SIGMA_POINTS>; // Matrix for sigma points of a

  // Parameters for UKF
  static constexpr double ALPHA  = alpha;
  static constexpr double BETA   = beta;
  static constexpr double KAPPA  = kappa;
  static constexpr double LAMBDA = ALPHA * ALPHA * (N_DIM_a + KAPPA) - N_DIM_a; // Parameter for spread of sigma points
  static constexpr double GAMMA  = std::sqrt(N_DIM_a + LAMBDA);                 // Scaling factor for spread of sigma points

  static constexpr std::array<double, N_SIGMA_POINTS> W_x = []() {
    std::array<double, N_SIGMA_POINTS> W_x;
    W_x[0] = LAMBDA / (N_DIM_a + LAMBDA);
    for (size_t i = 1; i < N_SIGMA_POINTS; i++) {
      W_x[i] = 1 / (2 * (N_DIM_a + LAMBDA));
    }
    return W_x;
  }();

  static constexpr std::array<double, N_SIGMA_POINTS> W_c = []() {
    std::array<double, N_SIGMA_POINTS> W_c;
    W_c[0] = LAMBDA / (N_DIM_a + LAMBDA) + (1 - ALPHA * ALPHA + BETA);
    for (size_t i = 1; i < N_SIGMA_POINTS; i++) {
      W_c[i] = 1 / (2 * (N_DIM_a + LAMBDA));
    }
    return W_c;
  }();

  UKF() = delete;

private:
  /** Get sigma points
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est T::Gauss_x State estimate
   * @return Mat_a2ap1 sigma_points
   */
  static Mat_a2ap1 get_sigma_points(const DynModT &dyn_mod, const SensModT &sens_mod, double dt, const T::Gauss_x &x_est)
  {
    typename T::Mat_xx P = x_est.cov();
    typename T::Mat_vv Q = dyn_mod.Q_d(dt, x_est.mean());
    typename T::Mat_ww R = sens_mod.R(x_est.mean());
    // Make augmented covariance matrix
    Mat_aa P_a = Mat_aa::Zero();
    /*
      P_a = | P  0  0 |
            | 0  Q  0 |
            | 0  0  R |
    */
    P_a.template topLeftCorner<N_DIM_x, N_DIM_x>()         = P;
    P_a.template block<N_DIM_v, N_DIM_v>(N_DIM_x, N_DIM_x) = Q;
    P_a.template bottomRightCorner<N_DIM_w, N_DIM_w>()     = R;

    Mat_aa sqrt_P_a = P_a.llt().matrixLLT();

    // Make augmented state vector
    Vec_a x_a;
    x_a << x_est.mean(), T::Vec_v::Zero(), T::Vec_w::Zero();

    // Calculate sigma points using the symmetric sigma point set
    Mat_a2ap1 sigma_points;
    sigma_points.col(0) = x_a;
    for (int i = 1; i <= N_DIM_a; i++) {
      sigma_points.col(i)           = x_a + GAMMA * sqrt_P_a.col(i - 1);
      sigma_points.col(i + N_DIM_a) = x_a - GAMMA * sqrt_P_a.col(i - 1);
    }
    return sigma_points;
  }

  /** Propagate sigma points through f
   * @param dyn_mod Dynamic model
   * @param dt Time step
   * @param sigma_points Mat_a2ap1 Sigma points
   * @param u T::Vec_u Control input (default 0)
   * @return Mat_x2ap1 sigma_x_pred
   */
  static Mat_x2ap1 propagate_sigma_points_f(const DynModT &dyn_mod, double dt, const Mat_a2ap1 &sigma_points, const T::Vec_u &u = T::Vec_u::Zero())
  {
    Eigen::Matrix<double, N_DIM_x, N_SIGMA_POINTS> sigma_x_pred;
    for (size_t i = 0; i < N_SIGMA_POINTS; i++) {
      typename T::Vec_x x_i = sigma_points.template block<N_DIM_x, 1>(0, i);
      typename T::Vec_v v_i = sigma_points.template block<N_DIM_v, 1>(N_DIM_x, i);
      sigma_x_pred.col(i) = dyn_mod.f_d(dt, x_i, u, v_i);
    }
    return sigma_x_pred;
  }

  /** Propagate sigma points through h
   * @param sens_mod Sensor model
   * @param sigma_points Mat_a2ap1 Sigma points
   * @return Mat_z2ap1 sigma_z_pred
   */
  static Mat_z2ap1 propagate_sigma_points_h(const SensModT &sens_mod, const Mat_a2ap1 &sigma_points)
  {
    Mat_z2ap1 sigma_z_pred;
    for (size_t i = 0; i < N_SIGMA_POINTS; i++) {
      typename T::Vec_x x_i = sigma_points.template block<N_DIM_x, 1>(0, i);
      typename T::Vec_w w_i = sigma_points.template block<N_DIM_w, 1>(N_DIM_x + N_DIM_v, i);
      sigma_z_pred.col(i) = sens_mod.h(x_i, w_i);
    }
    return sigma_z_pred;
  }

  /** Estimate gaussian from sigma points using the unscented transform
   * @param sigma_points Mat_n2ap1 Sigma points
   * @tparam n_dims Dimension of the gaussian
   * @return prob::MultiVarGauss<n_dims>
   * @note This function is templated to allow for different dimensions of the gaussian
   */
  template <int n_dims> static prob::MultiVarGauss<n_dims> estimate_gaussian(const Eigen::Matrix<double, n_dims, N_SIGMA_POINTS> &sigma_points)
  {
    using Vec_n  = Eigen::Vector<double, n_dims>;
    using Mat_nn = Eigen::Matrix<double, n_dims, n_dims>;

    // Predicted State Estimate x_k-
    Vec_n mean = Vec_n::Zero();
    for (size_t i = 0; i < N_SIGMA_POINTS; i++) {
      mean += W_x.at(i) * sigma_points.col(i);
    }
    Mat_nn cov = Mat_nn::Zero();
    for (size_t i = 0; i < N_SIGMA_POINTS; i++) {
      cov += W_c.at(i) * (sigma_points.col(i) - mean) * (sigma_points.col(i) - mean).transpose();
    }
    return {mean, cov};
  }

public:
  /** Perform one UKF prediction step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Previous state estimate
   * @param u T::Vec_u Control input (default 0)
   * @return std::pair<T::Gauss_x, T::Gauss_z> Predicted state estimate, predicted measurement estimate
   */
  static std::pair<typename T::Gauss_x, typename T::Gauss_z> predict(const DynModT &dyn_mod, const SensModT &sens_mod, double dt, const T::Gauss_x &x_est_prev,
                                                                     const T::Vec_u &u = T::Vec_u::Zero())
  {
    Mat_a2ap1 sigma_points = get_sigma_points(dyn_mod, sens_mod, dt, x_est_prev);

    // Propagate sigma points through f and h
    Mat_x2ap1 sigma_x_pred = propagate_sigma_points_f(dyn_mod, dt, sigma_points, u);
    Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sens_mod, sigma_points);

    // Predicted State and Measurement Estimate x_k- and z_k-
    typename T::Gauss_x x_pred = estimate_gaussian<N_DIM_x>(sigma_x_pred);
    typename T::Gauss_z z_pred = estimate_gaussian<N_DIM_z>(sigma_z_pred);

    return {x_pred, z_pred};
  }

  /** Perform one UKF update step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param x_est_pred Predicted state estimate
   * @param z_est_pred Predicted measurement estimate
   * @param z_meas Measurement
   * @return T::Gauss_x Updated state estimate
   * @note Sigma points are generated from the predicted state estimate instead of the previous state estimate as is done in the 'step' method.
   */
  static T::Gauss_x update(const DynModT &dyn_mod, const SensModT &sens_mod, double dt, const T::Gauss_x &x_est_pred, const T::Gauss_z &z_est_pred,
                           const T::Vec_z &z_meas)
  {
    // Generate sigma points from the predicted state estimate
    Mat_a2ap1 sigma_points = get_sigma_points(dyn_mod, sens_mod, dt, x_est_pred);

    // Extract the sigma points for the state
    Mat_x2ap1 sigma_x_pred = sigma_points.template block<N_DIM_x, N_SIGMA_POINTS>(0, 0);

    // Predict measurement
    Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sens_mod, sigma_points);

    // Calculate cross-covariance
    typename T::Mat_xz P_xz = T::Mat_xz::Zero();
    for (size_t i = 0; i < N_SIGMA_POINTS; i++) {
      P_xz += W_c(i) * (sigma_x_pred.col(i) - x_est_pred.mean()) * (sigma_z_pred.col(i) - z_est_pred.mean()).transpose();
    }

    // Calculate Kalman gain
    typename T::Mat_zz P_zz = z_est_pred.cov();
    typename T::Mat_xz K    = P_xz * P_zz.llt().solve(T::Mat_zz::Identity());

    // Update state estimate
    typename T::Vec_x x_upd_mean  = x_est_pred.mean() + K * (z_meas - z_est_pred.mean());
    typename T::Mat_xx x_upd_cov  = x_est_pred.cov() - K * P_zz * K.transpose();
    typename T::Gauss_x x_est_upd = {x_upd_mean, x_upd_cov};

    return x_est_upd;
  }

  /** Perform one UKF prediction and update step
   * @param dyn_mod Dynamic model
   * @param sens_mod Sensor model
   * @param dt Time step
   * @param x_est_prev Previous state estimate
   * @param z_meas Measurement
   * @param u T::Vec_u Control input
   * @return std::tuple<T::Gauss_x, T::Gauss_x, T::Gauss_z> Updated state estimate, predicted state estimate, predicted measurement estimate
   */
  static std::tuple<typename T::Gauss_x, typename T::Gauss_x, typename T::Gauss_z> step(const DynModT &dyn_mod, const SensModT &sens_mod, double dt,
                                                                                        const T::Gauss_x &x_est_prev, const T::Vec_z &z_meas, const T::Vec_u &u)
  {
    Mat_a2ap1 sigma_points = get_sigma_points(dyn_mod, sens_mod, dt, x_est_prev);

    // Propagate sigma points through f and h
    Mat_x2ap1 sigma_x_pred = propagate_sigma_points_f(dyn_mod, dt, sigma_points, u);
    Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sens_mod, sigma_points);

    // Predicted State and Measurement Estimate x_k- and z_k-
    typename T::Gauss_x x_pred = estimate_gaussian<N_DIM_x>(sigma_x_pred);
    typename T::Gauss_z z_pred = estimate_gaussian<N_DIM_z>(sigma_z_pred);

    // Calculate cross-covariance
    typename T::Mat_xz P_xz = T::Mat_xz::Zero();
    for (size_t i = 0; i < N_SIGMA_POINTS; i++) {
      P_xz += W_c.at(i) * (sigma_x_pred.col(i) - x_pred.mean()) * (sigma_z_pred.col(i) - z_pred.mean()).transpose();
    }

    // Calculate Kalman gain
    typename T::Mat_zz P_zz = z_pred.cov();
    typename T::Mat_xz K    = P_xz * P_zz.llt().solve(T::Mat_zz::Identity());

    // Update state estimate
    typename T::Vec_x x_upd_mean  = x_pred.mean() + K * (z_meas - z_pred.mean());
    typename T::Mat_xx x_upd_cov  = x_pred.cov() - K * P_zz * K.transpose();
    typename T::Gauss_x x_est_upd = {x_upd_mean, x_upd_cov};

    return {x_est_upd, x_pred, z_pred};
  }

  [[deprecated("use const DynModT &and const SensModT &")]] static std::pair<typename T::Gauss_x, typename T::Gauss_z>
  predict(std::shared_ptr<DynModT> dyn_mod, std::shared_ptr<SensModT> sens_mod, double dt, const T::Gauss_x &x_est_prev, const T::Vec_u &u = T::Vec_u::Zero())
  {
    return predict(*dyn_mod, *sens_mod, dt, x_est_prev, u);
  }

  [[deprecated("use const DynModT &and const SensModT &")]] static T::Gauss_x update(std::shared_ptr<DynModT> dyn_mod, std::shared_ptr<SensModT> sens_mod,
                                                                                     const T::Gauss_x &x_est_pred, const T::Gauss_z &z_est_pred,
                                                                                     const T::Vec_z &z_meas)
  {
    return update(*dyn_mod, *sens_mod, x_est_pred, z_est_pred, z_meas);
  }

  [[deprecated("use const DynModT &and const SensModT &")]] static std::tuple<typename T::Gauss_x, typename T::Gauss_x, typename T::Gauss_z>
  step(std::shared_ptr<DynModT> dyn_mod, std::shared_ptr<SensModT> sens_mod, double dt, const T::Gauss_x &x_est_prev, const T::Vec_z &z_meas, const T::Vec_u &u)
  {
    return step(*dyn_mod, *sens_mod, dt, x_est_prev, z_meas, u);
  }
};

} // namespace filter
} // namespace vortex