/**
 * @file EKF.hpp
 * @author Eirik Kolås
 * @brief 
 * @version 0.1
 * @date 2023-10-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once
#include <tuple>
#include <probability/multi_var_gauss.hpp>

namespace vortex {
namespace filters {

/** @brief Extended Kalman Filter
 * 
 * @tparam DynamicModel Dynamic model type. Has to have function for Jacobian of state transition.
 * @tparam SensorModel Sensor model type. Has to have function for Jacobian of measurement. (get_H)
 */
template <class DynamicModel, class SensorModel>
class EKF {
public:
    EKF(DynamicModel dynamic_model, SensorModel sensor_model)
        : dynamic_model_(dynamic_model), sensor_model_(sensor_model) {}

    /** Perform one EKF prediction step
     * @param x_est_prev Previous state estimate
     * @param dt Time step
    */
    std::tuple<DynamicModel, SensorModel> predict(std::chrono::duration<double> dt, prob::MultiVarGauss<n_dim_x_> x_est_prev) {
        prob::MultiVarGauss<n_dim_x_> x_est_pred = dynamic_model_.pred_from_est(x_est_prev, dt);
        prob::MultiVarGauss<n_dim_z_> z_est_pred = sensor_model_.pred_from_est(x_est_pred);
        return {x_est_pred, z_est_pred};
    }

    /** Perform one EKF update step
     * @param x_est_pred Predicted state
     * @param z_est_pred Predicted measurement
     * @param z_meas Measurement
     * @return MultivarGauss Updated state
    */
    prob::MultiVarGauss<n_dim_x_> update(prob::MultiVarGauss<n_dim_x_> x_est_pred, prob::MultiVarGauss<n_dim_z_> z_est_pred, Eigen::Vector<double, n_dim_z_> z_meas) {
        auto H_mat = sensor_model_.H(x_est_pred);
        auto R_mat = sensor_model_.R(x_est_pred);
        auto P_mat = x_est_pred.cov();
        auto S_mat = z_est_pred.cov();
        auto S_mat_inv = z_est_pred.cov_inv();
        auto I = Eigen::Matrix<double, n_dim_x_, n_dim_x_>::Identity();

        Eigen::Matrix<double, n_dim_x_, n_dim_z_> kalman_gain = P_mat * H_mat.transpose() * S_mat_inv;
        Eigen::Vector<double, n_dim_z_> innovation = z_meas - z_est_pred.mean();

        Eigen::Matrix<double, n_dim_x_, n_dim_x_> state_upd_mean = x_est_pred.mean() + kalman_gain * innovation;
        // Use the Joseph form of the covariance update to ensure positive definiteness
        Eigen::Matrix<double, n_dim_x_, n_dim_x_> state_upd_cov = (I - kalman_gain * H_mat) * P_mat * (I - kalman_gain * H_mat).transpose() + kalman_gain * R_mat * kalman_gain.transpose();

        return prob::MultiVarGauss<n_dim_x_>(state_upd_mean, state_upd_cov);
    }

private:
    const Dynamic_model dynamic_model_;
    const Sensor_model sensor_model_;
    static constexpr n_dim_x_ = DynamicModel::n_dim;
    static constexpr n_dim_z_ = SensorModel::n_dim;
};

}  // namespace filters
}  // namespace vortex