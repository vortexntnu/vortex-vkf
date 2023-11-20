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
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

namespace vortex {
namespace filter {

/** @brief Extended Kalman Filter. (I stands for interface, T for Type)
 * 
 * @tparam n_dim_x State dimension.
 * @tparam n_dim_z Measurement dimension.
 * @tparam n_dim_u Input dimension (Default: n_dim_x)
 * @tparam n_dim_v Process noise dimension (Default: n_dim_x)
 * @tparam n_dim_w Measurement noise dimension (Default: n_dim_z)
 */
template <int n_dim_x, int n_dim_z, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x, int n_dim_w = n_dim_z>
class EKF : public interface::KalmanFilter<n_dim_x, n_dim_z, n_dim_u, n_dim_v, n_dim_w> {
public:
    static constexpr int N_DIM_x = n_dim_x;
    static constexpr int N_DIM_u = n_dim_u;
    static constexpr int N_DIM_z = n_dim_z;
    static constexpr int N_DIM_v = n_dim_v;
    static constexpr int N_DIM_w = n_dim_w;

    using DynModI  = models::DynamicModelI<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModI = models::SensorModelI<N_DIM_x, N_DIM_z, N_DIM_w>;
    using DynModIPtr = std::shared_ptr<DynModI>;
    using SensModIPtr = std::shared_ptr<SensModI>;

    using DynModEKF  = models::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModEKF = models::SensorModelLTV<N_DIM_x, N_DIM_z>;
    using DynModEKFPtr = std::shared_ptr<DynModEKF>;
    using SensModEKFPtr = std::shared_ptr<SensModEKF>;

    // TODO: Clean up typedefs
    using Vec_x    = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx   = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Vec_z    = Eigen::Vector<double, N_DIM_z>;
    using Mat_zz   = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
    using Mat_zx   = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_xz   = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
    using Gauss_x  = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_z  = prob::MultiVarGauss<N_DIM_z>;

    EKF(DynModEKFPtr dynamic_model, SensModEKFPtr sensor_model)
        : dynamic_model_(dynamic_model), sensor_model_(sensor_model) {}

    /** Perform one EKF prediction step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @param u Vec_x Input. Not used, set to zero.
     * @return std::pair<Gauss_x, Gauss_z> Predicted state, predicted measurement
     * @throws std::runtime_error if dyn_mod or sens_mod are not of the DynamicModelT or SensorModelT type
     * @note Overridden from interface::KalmanFilter
    */
    std::pair<Gauss_x, Gauss_z> predict(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x& x_est_prev, const Vec_x& = Vec_x::Zero()) const override
    {
        // cast to dynamic model type to access pred_from_est
        auto dyn_model = std::dynamic_pointer_cast<DynModEKF>(dyn_mod);
        // cast to sensor model type to access pred_from_est
        auto sens_model = std::dynamic_pointer_cast<SensModEKF>(sens_mod);

        Gauss_x x_est_pred = dyn_model->pred_from_est(dt, x_est_prev);
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
     * @note Overridden from interface::KalmanFilter
    */
    Gauss_x update(DynModIPtr, SensModIPtr sens_mod, const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) const override
    {
        // cast to sensor model type
        auto sens_model = std::dynamic_pointer_cast<SensModEKF>(sens_mod);
        Mat_zx C = sens_model->C(x_est_pred.mean());
        Mat_zz R = sens_model->R(x_est_pred.mean());
        Mat_xx P = x_est_pred.cov();
        Mat_zz S_inv = z_est_pred.cov_inv();
        Mat_xx I = Mat_xx::Identity(N_DIM_x, N_DIM_x);

        Mat_xz kalman_gain = P * C.transpose() * S_inv;
        Vec_z innovation = z_meas - z_est_pred.mean();

        Vec_x state_upd_mean = x_est_pred.mean() + kalman_gain * innovation;
        // Use the Joseph form of the covariance update to ensure positive definiteness
        Mat_xx state_upd_cov = (I - kalman_gain * C) * P * (I - kalman_gain * C).transpose() + kalman_gain * R * kalman_gain.transpose();

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
     * @note Overridden from interface::KalmanFilter
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x& x_est_prev, const Vec_z& z_meas, const Vec_x& u) const override
    {
        std::pair<Gauss_x, Gauss_z> pred = predict(dyn_mod, sens_mod, dt, x_est_prev, u);
        Gauss_x x_est_pred = pred.first;
        Gauss_z z_est_pred = pred.second;
        Gauss_x x_est_upd  = update(dyn_mod, sens_mod, x_est_pred, z_est_pred, z_meas);
        return {x_est_upd, x_est_pred, z_est_pred};
    }

    /** Perform one EKF prediction step
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @return Predicted state, predicted measurement
    */
    std::pair<Gauss_x, Gauss_z> predict(double dt, const Gauss_x& x_est_prev) const {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        return predict(dynamic_model_, sensor_model_, dt, x_est_prev, Vec_x::Zero());
    }

    /** Perform one EKF update step
     * @param x_est_pred Predicted state
     * @param z_est_pred Predicted measurement
     * @param z_meas Vec_z
     * @return MultivarGauss Updated state
    */
    Gauss_x update(const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) const {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        return update(dynamic_model_, sensor_model_, x_est_pred, z_est_pred, z_meas);
    }

    /** Perform one EKF prediction and update step
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @param z_meas Vec_z
     * @return Updated state, predicted state, predicted measurement
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(double dt, const Gauss_x& x_est_prev, const Vec_z& z_meas) const {
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("Dynamic model or sensor model not set");
        }
        return step(dynamic_model_, sensor_model_, dt, x_est_prev, z_meas, Vec_x::Zero());
    }

private:
    const DynModEKFPtr dynamic_model_;
    const SensModEKFPtr sensor_model_;
};

/** @brief EKF with dimensions defined by the dynamic model and sensor model.
 * 
 * @tparam DynModT Dynamic model type.
 * @tparam SensModT Sensor model type. 
 */
template <typename DynModT, typename SensModT>
using EKF_M = EKF<DynModT::N_DIM_x, SensModT::N_DIM_z, DynModT::N_DIM_u, DynModT::N_DIM_v, SensModT::N_DIM_w>;

}  // namespace filter
}  // namespace vortex