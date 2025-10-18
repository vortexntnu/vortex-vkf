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
#include <type_traits>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::filter {

/**
 * @brief Extended Kalman Filter
 *
 * @tparam n_dim_x State dimension
 * @tparam n_dim_z Measurement dimension
 * @tparam n_dim_u Input dimension
 * @tparam n_dim_v Process noise dimension
 * @tparam n_dim_w Measurement noise dimension
 */
template <size_t n_dim_x,
          size_t n_dim_z,
          size_t n_dim_u = n_dim_x,
          size_t n_dim_v = n_dim_x,
          size_t n_dim_w = n_dim_z>
class EKF_t {
   public:
    static constexpr int N_DIM_x = (int)n_dim_x;
    static constexpr int N_DIM_z = (int)n_dim_z;
    static constexpr int N_DIM_u = (int)n_dim_u;
    static constexpr int N_DIM_v = (int)n_dim_v;
    static constexpr int N_DIM_w = (int)n_dim_w;

    using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

    EKF_t() = delete;

    /** Perform one EKF prediction step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param dt Time step
     * @param x_est_prev Gauss_x Previous state estimate
     * @param u Vec_x Input. Not used, set to zero.
     * @return std::pair<Gauss_x, Gauss_z> Predicted state, predicted
     * measurement
     * @throws std::runtime_error if dyn_mod or sens_mod are not of the
     * DynamicModelT or SensorModelT type
     */
    static auto predict(const auto& dyn_mod,
                        const auto& sens_mod,
                        double dt,
                        const auto& x_est_prev,
                        const T::Vec_u& u = T::Vec_u::Zero())
        -> std::pair<std::remove_reference_t<decltype(x_est_prev)>,
                     typename T::Gauss_z>
        requires(
            concepts::
                DynamicModelLTV<decltype(dyn_mod), N_DIM_x, N_DIM_u, N_DIM_v> &&
            concepts::
                SensorModelLTV<decltype(sens_mod), N_DIM_x, N_DIM_z, N_DIM_w> &&
            concepts::MultiVarGaussLike<decltype(x_est_prev), N_DIM_x>)
    {
        using StateT = std::remove_reference_t<decltype(x_est_prev)>;
        StateT x_est_pred = StateT{dyn_mod.pred_from_est(dt, x_est_prev, u)};
        typename T::Gauss_z z_est_pred = sens_mod.pred_from_est(x_est_pred);
        return {x_est_pred, z_est_pred};
    }

    /** Perform one EKF update step
     * @param sens_mod Sensor model
     * @param x_est_pred Gauss_x Predicted state
     * @param z_est_pred Gauss_z Predicted measurement
     * @param z_meas Vec_z Measurement
     * @return MultivarGauss Updated state
     * @throws std::runtime_error ifsens_mod is not of the SensorModelT type
     */
    static auto update(const auto& sens_mod,
                       const auto& x_est_pred,
                       const auto& z_est_pred,
                       const T::Vec_z& z_meas)
        -> std::remove_reference_t<decltype(x_est_pred)>
        requires(
            concepts::
                SensorModelLTV<decltype(sens_mod), N_DIM_x, N_DIM_z, N_DIM_w> &&
            concepts::MultiVarGaussLike<decltype(x_est_pred), N_DIM_x> &&
            concepts::MultiVarGaussLike<decltype(z_est_pred), N_DIM_z>)
    {
        typename T::Mat_zx C =
            sens_mod.C(x_est_pred.mean());  // Measurement matrix
        typename T::Mat_ww R =
            sens_mod.R(x_est_pred.mean());  // Measurement noise covariance
        typename T::Mat_zw H = sens_mod.H(
            x_est_pred.mean());  // Measurement noise cross-covariance
        typename T::Mat_xx P = x_est_pred.cov();  // State covariance
        typename T::Mat_zz S_inv =
            z_est_pred
                .cov_inv();  // Inverse of the predicted measurement covariance
        typename T::Mat_xx I = T::Mat_xx::Identity(N_DIM_x, N_DIM_x);

        typename T::Mat_xz W = P * C.transpose() * S_inv;  // Kalman gain
        typename T::Vec_z innovation = z_meas - z_est_pred.mean();

        // If the sensor model supports angular wrapping, apply it
        if constexpr (requires { sens_mod.wrap_residual(innovation); }) {
            innovation = sens_mod.wrap_residual(innovation);
        }

        typename T::Vec_x state_upd_mean = x_est_pred.mean() + W * innovation;
        // Use the Joseph form of the covariance update to ensure positive
        // definiteness
        typename T::Mat_xx state_upd_cov =
            (I - W * C) * P * (I - W * C).transpose() +
            W * H * R * H.transpose() * W.transpose();

        return {state_upd_mean, state_upd_cov};
    }

    /** Perform one EKF prediction and update step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param dt Time step
     * @param x_est_prev Gauss_x Previous state estimate
     * @param z_meas Vec_z Measurement
     * @param u Vec_x Input
     * @return Updated state, predicted state, predicted measurement
     */
    static auto step(const auto& dyn_mod,
                     const auto& sens_mod,
                     double dt,
                     const auto& x_est_prev,
                     const T::Vec_z& z_meas,
                     const T::Vec_u& u = T::Vec_u::Zero())
        -> std::tuple<std::remove_reference_t<decltype(x_est_prev)>,
                      std::remove_reference_t<decltype(x_est_prev)>,
                      typename T::Gauss_z>
        requires(
            concepts::
                DynamicModelLTV<decltype(dyn_mod), N_DIM_x, N_DIM_u, N_DIM_v> &&
            concepts::
                SensorModelLTV<decltype(sens_mod), N_DIM_x, N_DIM_z, N_DIM_w> &&
            concepts::MultiVarGaussLike<decltype(x_est_prev), N_DIM_x>)
    {
        auto [x_est_pred, z_est_pred] =
            predict(dyn_mod, sens_mod, dt, x_est_prev, u);

        auto x_est_upd = update(sens_mod, x_est_pred, z_est_pred, z_meas);
        return {x_est_upd, x_est_pred, z_est_pred};
    }
};

/**
 * @brief Extended Kalman Filter.
 * @tparam DynModT Dynamic model type derived from
 * `vortex::models::interface::DynamicModelLTV`
 * @tparam SensModT Sensor model type derived from
 * `vortex::models::interface::SensorModelLTV`
 */
template <concepts::DynamicModelLTVWithDefinedSizes DynModT,
          concepts::SensorModelLTVWithDefinedSizes SensModT>
using EKF = EKF_t<DynModT::N_DIM_x,
                  SensModT::N_DIM_z,
                  DynModT::N_DIM_u,
                  DynModT::N_DIM_v,
                  SensModT::N_DIM_w>;

}  // namespace vortex::filter
