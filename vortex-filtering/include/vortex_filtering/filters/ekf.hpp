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
#include <vortex_filtering/filters/filter_base.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/sensor_model.hpp>

namespace vortex {
namespace filters {

/** @brief Extended Kalman Filter. (I stands for interface, T for Type)
 * 
 * @tparam DynamicModelEKFI Dynamic model type. Has to have function for Jacobian of state transition.
 * @tparam SensorModelI Sensor model type. Has to have function for Jacobian of measurement. (get_H)
 */
template <class DynModT, class SensModT>
class EKF : public KalmanFilterI<DynModT, SensModT> {
public:
    static constexpr int N_DIM_x = DynModT::N_DIM_x;
    static constexpr int N_DIM_z = SensModT::N_DIM_z;
    static constexpr int N_DIM_u = DynModT::N_DIM_u;
    static constexpr int N_DIM_v = DynModT::N_DIM_v;
    static constexpr int N_DIM_w = SensModT::N_DIM_w;

    using DynModI  = models::DynamicModelI<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModI = models::SensorModelI<N_DIM_x, N_DIM_z, N_DIM_w>;
    using DynModIShared = std::shared_ptr<DynModI>;
    using SensModIShared = std::shared_ptr<SensModI>;

    using Vec_x    = typename Eigen::Vector<double, N_DIM_x>;
    using Mat_xx   = typename Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Vec_z    = typename Eigen::Vector<double, N_DIM_z>;
    using Mat_zz   = typename Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
    using Mat_zx   = typename Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_xz   = typename Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
    using Gauss_x  = typename prob::MultiVarGauss<N_DIM_x>;
    using Gauss_z  = typename prob::MultiVarGauss<N_DIM_z>;

    EKF(std::shared_ptr<DynModT> dynamic_model, std::shared_ptr<SensModT> sensor_model)
        : dynamic_model_(dynamic_model), sensor_model_(sensor_model) {}

    /** Perform one EKF prediction step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param x_est_prev Previous state estimate
     * @param dt Time step
     * @return std::pair<Gauss_x, Gauss_z> Predicted state, predicted measurement
     * @throws std::runtime_error if dyn_mod or sens_mod are not of the DynamicModelT or SensorModelT type
    */
    std::pair<Gauss_x, Gauss_z> predict(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_prev, const Vec_x&, double dt) override
    {
        // cast to dynamic model type to access pred_from_est
        auto dyn_model = std::dynamic_pointer_cast<DynModT>(dyn_mod);
        // cast to sensor model type to access pred_from_est
        auto sens_model = std::dynamic_pointer_cast<SensModT>(sens_mod);

        Gauss_x x_est_pred = dyn_model->pred_from_est(x_est_prev, dt);
        Gauss_z z_est_pred = sens_model->pred_from_est(x_est_pred);
        return {x_est_pred, z_est_pred};
    }

    /** Perform one EKF update step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param x_est_pred Predicted state
     * @param z_est_pred Predicted measurement
     * @param z_meas Vec_z Measurement
     * @return MultivarGauss Updated state
     * @throws std::runtime_error ifsens_mod is not of the SensorModelT type
    */
    Gauss_x update(DynModIShared, SensModIShared sens_mod, const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) override
    {
        // cast to sensor model type
        auto sens_model = std::dynamic_pointer_cast<SensModT>(sens_mod);
        Mat_zx H_mat = sens_model->H(x_est_pred.mean());
        Mat_zz R_mat = sens_model->R(x_est_pred.mean());
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

    /** Perform one EKF prediction and update step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param x_est_prev Previous state estimate
     * @param z_meas Vec_z Measurement
     * @param u Vec_x Input
     * @param dt Time step
     * @return Updated state, predicted state, predicted measurement
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_prev, const Vec_z& z_meas, const Vec_x& u, double dt) override
    {
        std::pair<Gauss_x, Gauss_z> pred = predict(dyn_mod, sens_mod, x_est_prev, u, dt);
        Gauss_x x_est_pred = pred.first;
        Gauss_z z_est_pred = pred.second;
        Gauss_x x_est_upd  = update(dyn_mod, sens_mod, x_est_pred, z_est_pred, z_meas);
        return {x_est_upd, x_est_pred, z_est_pred};
    }

    /** Perform one EKF prediction step
     * @param x_est_prev Previous state estimate
     * @param dt Time step
     * @return Predicted state, predicted measurement
    */
    std::pair<Gauss_x, Gauss_z> predict(const Gauss_x& x_est_prev, double dt) {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        return predict(dynamic_model_, sensor_model_, x_est_prev, Vec_x::Zero(), dt);
    }

    /** Perform one EKF update step
     * @param x_est_pred Predicted state
     * @param z_est_pred Predicted measurement
     * @param z_meas Vec_z
     * @return MultivarGauss Updated state
    */
    Gauss_x update(const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        return update(dynamic_model_, sensor_model_, x_est_pred, z_est_pred, z_meas);
    }

    /** Perform one EKF prediction and update step
     * @param x_est_prev Previous state estimate
     * @param z_meas Vec_z
     * @param dt Time step
     * @return Updated state, predicted state, predicted measurement
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(const Gauss_x& x_est_prev, const Vec_z& z_meas, double dt) {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        return step(dynamic_model_, sensor_model_, x_est_prev, z_meas, Vec_x::Zero(), dt);
    }

private:
    const DynModIShared dynamic_model_;
    const SensModIShared sensor_model_;
};

}  // namespace filters
}  // namespace vortex