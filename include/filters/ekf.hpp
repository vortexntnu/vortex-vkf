/**
 * @file ekf.hpp
 * @author Eirik Kolås
 * @brief Multivariate Gaussian Distribution. Based on "Fundamentals of Sensor Fusion" by Edmund Brekke
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
template <class DynamicModelT, class SensorModelT>
class EKF {
    static constexpr int N_DIM_x = DynamicModelT::N_DIM_x;
    static constexpr int N_DIM_z = SensorModelT::N_DIM_z;
public:
    EKF(DynamicModelT dynamic_model, SensorModelT sensor_model)
        : dynamic_model_(std::move(dynamic_model)), sensor_model_(std::move(sensor_model)) {}

    /** Perform one EKF prediction step
     * @param x_est_prev Previous state estimate
     * @param dt Time step
    */
    std::pair<prob::MultiVarGauss<N_DIM_x>, prob::MultiVarGauss<N_DIM_z>> predict(const prob::MultiVarGauss<N_DIM_x>& x_est_prev, double dt) {
        auto x_est_pred = dynamic_model_.pred_from_est(x_est_prev, dt);
        auto z_est_pred = sensor_model_.pred_from_est(x_est_pred);
        return {x_est_pred, z_est_pred};
    }

    /** Perform one EKF update step
     * @param x_est_pred Predicted state
     * @param z_est_pred Predicted measurement
     * @param z_meas Measurement
     * @return MultivarGauss Updated state
    */
    prob::MultiVarGauss<N_DIM_x> update(const prob::MultiVarGauss<N_DIM_x>& x_est_pred, const prob::MultiVarGauss<N_DIM_z>& z_est_pred, const Eigen::Vector<double, N_DIM_z>& z_meas) {
        auto H_mat = sensor_model_.H(x_est_pred.mean());
        auto R_mat = sensor_model_.R(x_est_pred.mean());
        auto P_mat = x_est_pred.cov();
        auto S_mat = z_est_pred.cov();
        auto S_mat_inv = z_est_pred.cov_inv();
        auto I = Eigen::Matrix<double, N_DIM_x, N_DIM_x>::Identity();

        Eigen::Matrix<double, N_DIM_x, N_DIM_z> kalman_gain = P_mat * H_mat.transpose() * S_mat_inv;
        Eigen::Vector<double, N_DIM_z> innovation = z_meas - z_est_pred.mean();

        Eigen::Matrix<double, N_DIM_x, N_DIM_x> state_upd_mean = x_est_pred.mean() + kalman_gain * innovation;
        // Use the Joseph form of the covariance update to ensure positive definiteness
        Eigen::Matrix<double, N_DIM_x, N_DIM_x> state_upd_cov = (I - kalman_gain * H_mat) * P_mat * (I - kalman_gain * H_mat).transpose() + kalman_gain * R_mat * kalman_gain.transpose();

        return prob::MultiVarGauss<N_DIM_x>(state_upd_mean, state_upd_cov);
    }

    /**
     * @brief Perform one EKF update step
     * @param x_est_prev Previous state estimate
     * @param z_meas Measurement
     * @param dt Time step
     */
    std::tuple<prob::MultiVarGauss<N_DIM_x>, prob::MultiVarGauss<N_DIM_x>, prob::MultiVarGauss<N_DIM_z>> step(const prob::MultiVarGauss<N_DIM_x>& x_est_prev, const Eigen::Vector<double, N_DIM_z>& z_meas, double dt) {
        auto [x_est_pred, z_est_pred] = predict(dt, x_est_prev);
        auto x_est_upd = update(x_est_pred, z_est_pred, z_meas);
        return {x_est_upd, x_est_pred, z_est_pred};
    }

private:
    const DynamicModelT dynamic_model_;
    const SensorModelT sensor_model_;
};

}  // namespace filters
}  // namespace vortex