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
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/sensor_model.hpp>

namespace vortex {
namespace filters {

/** @brief Extended Kalman Filter. (I stands for interface, T for Type)
 * 
 * @tparam DynamicModelI Dynamic model type. Has to have function for Jacobian of state transition.
 * @tparam SensorModelI Sensor model type. Has to have function for Jacobian of measurement. (get_H)
 */
template <class DynamicModelT, class SensorModelT>
class EKF {
public:
    static constexpr int N_DIM_x = DynamicModelT::N_DIM_x;
    static constexpr int N_DIM_z = SensorModelT::N_DIM_z;
    using DynModI  = models::DynamicModelI<N_DIM_x>;
    using SensModI = models::SensorModelI<N_DIM_x, N_DIM_z>;
    using Vec_x    = typename DynModI::Vec_x;
    using Mat_xx   = typename DynModI::Mat_xx;
    using Vec_z    = typename SensModI::Vec_z;
    using Mat_zz   = typename SensModI::Mat_zz;
    using Mat_zx   = typename SensModI::Mat_zx;
    using Mat_xz   = typename SensModI::Mat_xz;
    using Gauss_x  = typename DynModI::Gauss_x;
    using Gauss_z  = typename SensModI::Gauss_z;

    EKF(DynamicModelT dynamic_model, SensorModelT sensor_model)
        : dynamic_model_(std::move(dynamic_model)), sensor_model_(std::move(sensor_model)) {}

    /** Perform one EKF prediction step
     * @param x_est_prev Previous state estimate
     * @param dt Time step
    */
    std::pair<Gauss_x, Gauss_z> predict(const Gauss_x& x_est_prev, double dt) {
        Gauss_x x_est_pred = dynamic_model_.pred_from_est(x_est_prev, dt);
        Gauss_z z_est_pred = sensor_model_.pred_from_est(x_est_pred);
        return {x_est_pred, z_est_pred};
    }

    /** Perform one EKF update step
     * @param x_est_pred Predicted state
     * @param z_est_pred Predicted measurement
     * @param z_meas Vec_z
     * @return MultivarGauss Updated state
    */
    Gauss_x update(const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) {
        Mat_zx H_mat = sensor_model_.H(x_est_pred.mean());
        Mat_zz R_mat = sensor_model_.R(x_est_pred.mean());
        Mat_xx P_mat = x_est_pred.cov();
        Mat_zz S_mat_inv = z_est_pred.cov_inv();
        Mat_xx I = Mat_xx::Identity(N_DIM_x, N_DIM_x);

        Mat_xz kalman_gain = P_mat * H_mat.transpose() * S_mat_inv;
        Vec_z innovation = z_meas - z_est_pred.mean();

        Vec_x state_upd_mean = x_est_pred.mean() + kalman_gain * innovation;
        // Use the Joseph form of the covariance update to ensure positive definiteness
        Mat_xx state_upd_cov = (I - kalman_gain * H_mat) * P_mat * (I - kalman_gain * H_mat).transpose() + kalman_gain * R_mat * kalman_gain.transpose();

        return {state_upd_mean, state_upd_cov};
    }

    /**
     * @brief Perform one EKF prediction and update step
     * @param x_est_prev Previous state estimate
     * @param z_meas Vec_z
     * @param dt Time step
     * @return std::tuple<Gauss_x, Gauss_x, Gauss_z> Updated state, predicted state, predicted measurement
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(const Gauss_x& x_est_prev, const Vec_z& z_meas, double dt) {
        std::pair<Gauss_x, Gauss_z> pred = predict(x_est_prev, dt);
        Gauss_x x_est_pred = pred.first;
        Gauss_z z_est_pred = pred.second;
        Gauss_x x_est_upd  = update(x_est_pred, z_est_pred, z_meas);
        return {x_est_upd, x_est_pred, z_est_pred};
    }

private:
    const DynamicModelT dynamic_model_;
    const SensorModelT sensor_model_;
};

}  // namespace filters
}  // namespace vortex