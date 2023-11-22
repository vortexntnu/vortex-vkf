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
#include <vortex_filtering/filters/filter_base.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

namespace vortex {
namespace filter {

    /** Unscented Kalman Filter
     * @tparam n_dim_x Dimension of state vector
     * @tparam n_dim_z Dimension of measurement vector
     * @tparam n_dim_u Dimension of control input vector (default n_dim_x)
     * @tparam n_dim_v Dimension of process noise vector (default n_dim_x)
     * @tparam n_dim_w Dimension of measurement noise vector (default n_dim_z)
     */
template<int n_dim_x, int n_dim_z, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x, int n_dim_w = n_dim_z>
class UKF : public interface::KalmanFilter<n_dim_x, n_dim_z, n_dim_u, n_dim_v, n_dim_w> {
public:
    static constexpr int N_DIM_x = n_dim_x;
    static constexpr int N_DIM_u = n_dim_u;
    static constexpr int N_DIM_z = n_dim_z;
    static constexpr int N_DIM_v = n_dim_v;
    static constexpr int N_DIM_w = n_dim_w;

    using DynModI  = models::interface::DynamicModelI<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModI = models::interface::SensorModelI<N_DIM_x, N_DIM_z, N_DIM_w>;
    using DynModIPtr = std::shared_ptr<DynModI>;
    using SensModIPtr = std::shared_ptr<SensModI>;

    using Vec_x    = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx   = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;

    using Vec_u    = Eigen::Vector<double, DynModI::N_DIM_u>;

    using Vec_z    = Eigen::Vector<double, N_DIM_z>;
    using Mat_zz   = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
    using Mat_zx   = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_xz   = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;

    using Vec_v    = Eigen::Vector<double, N_DIM_v>;
    using Mat_vv   = Eigen::Matrix<double, N_DIM_v, N_DIM_v>;
    using Mat_xv   = Eigen::Matrix<double, N_DIM_x, N_DIM_v>;
    using Mat_vx   = Eigen::Matrix<double, N_DIM_v, N_DIM_x>;

    using Vec_w    = Eigen::Vector<double, N_DIM_w>;
    using Mat_xw   = Eigen::Matrix<double, N_DIM_x, N_DIM_w>;
    using Mat_wx   = Eigen::Matrix<double, N_DIM_w, N_DIM_x>;
    using Mat_zw   = Eigen::Matrix<double, N_DIM_z, N_DIM_w>;
    using Mat_ww   = Eigen::Matrix<double, N_DIM_w, N_DIM_w>;
    using Mat_vw   = Eigen::Matrix<double, N_DIM_v, N_DIM_w>;
    using Mat_wv   = Eigen::Matrix<double, N_DIM_w, N_DIM_v>;

    using Gauss_x  = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_z  = prob::MultiVarGauss<N_DIM_z>;

    static constexpr int N_DIM_a = N_DIM_x + N_DIM_v + N_DIM_w;         // Augmented state dimension
    using Vec_a     = Eigen::Vector<double, N_DIM_a>;                   // Augmented state vector
    using Mat_aa    = Eigen::Matrix<double, N_DIM_a, N_DIM_a>;          // Augmented state covariance matrix
    using Mat_x2ap1 = Eigen::Matrix<double, N_DIM_x, 2*N_DIM_a + 1>;    // Matrix for sigma points of x
    using Mat_z2ap1 = Eigen::Matrix<double, N_DIM_z, 2*N_DIM_a + 1>;    // Matrix for sigma points of z
    using Mat_a2ap1 = Eigen::Matrix<double, N_DIM_a, 2*N_DIM_a + 1>;    // Matrix for sigma points of a

      
    /** Unscented Kalman Filter
     * @param dynamic_model Dynamic model (default nullptr)
     * @param sensor_model Sensor model (default nullptr)
     * @param alpha Parameter for spread of sigma points (default 1.0)
     * @param beta Parameter for weighting of mean in covariance calculation (default 2.0)
     * @param kappa Parameter for adding additional spread to sigma points (default 0.0)
     * @tparam n_dim_x Dimension of state vector
     * @tparam n_dim_z Dimension of measurement vector
     * @tparam n_dim_u Dimension of control input vector (default n_dim_x)
     * @tparam n_dim_v Dimension of process noise vector (default n_dim_x)
     * @tparam n_dim_w Dimension of measurement noise vector (default n_dim_z)
     */
    UKF(DynModIPtr dynamic_model = nullptr, SensModIPtr sensor_model = nullptr, double alpha = 1.0, double beta = 2.0, double kappa = 0.0)
        : dynamic_model_(dynamic_model), sensor_model_(sensor_model)
        , ALPHA_(alpha), BETA_(beta), KAPPA_(kappa)
    {
        // Parameters for UKF
        LAMBDA_ = ALPHA_*ALPHA_*(N_DIM_a + KAPPA_) - N_DIM_a;
        GAMMA_ = std::sqrt(N_DIM_a + LAMBDA_);

        // Scaling factors
        W_x_.resize(2*N_DIM_a + 1);
        W_c_.resize(2*N_DIM_a + 1);
        W_x_[0] = LAMBDA_ / (N_DIM_a + LAMBDA_);
        W_c_[0] = LAMBDA_ / (N_DIM_a + LAMBDA_) + (1 - ALPHA_*ALPHA_ + BETA_);
        for (int i = 1; i < 2*N_DIM_a + 1; i++) {
            W_x_[i] = 1 / (2*(N_DIM_a + LAMBDA_));
            W_c_[i] = 1 / (2*(N_DIM_a + LAMBDA_));
        }
    }

private:
    /** Get sigma points
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param dt Time step
     * @param x_est Gauss_x State estimate
     * @return Mat_a2ap1 sigma_points
    */
    Mat_a2ap1 get_sigma_points(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x& x_est) const
    {
        Mat_xx P = x_est.cov();
        Mat_vv Q = dyn_mod->Q_d(dt, x_est.mean());
        Mat_ww R = sens_mod->R(x_est.mean());
        // Make augmented covariance matrix
        Mat_aa P_a;
        // clang-format off
        P_a << P	         , Mat_xv::Zero() , Mat_xw::Zero(),
               Mat_vx::Zero(), Q              , Mat_vw::Zero(),
               Mat_wx::Zero(), Mat_wv::Zero() , R;	
        // clang-format on
        Mat_aa sqrt_P_a = P_a.llt().matrixLLT();

        // Make augmented state vector
        Vec_a x_a;
        x_a << x_est.mean(), Vec_v::Zero(), Vec_w::Zero();

        // Calculate sigma points using the symmetric sigma point set
        Mat_a2ap1 sigma_points;
        sigma_points.col(0) = x_a;
        for (int i = 1; i <= N_DIM_a; i++) {
            sigma_points.col(i)           = x_a + GAMMA_ * sqrt_P_a.col(i - 1);
            sigma_points.col(i + N_DIM_a) = x_a - GAMMA_ * sqrt_P_a.col(i - 1);
        }
        return sigma_points;
    }

    /** Propagate sigma points through f
     * @param dyn_mod Dynamic model
     * @param dt Time step
     * @param sigma_points Mat_a2ap1 Sigma points
     * @param u Vec_u Control input
     * @return Mat_x2ap1 sigma_x_pred
     */
    Mat_x2ap1 propagate_sigma_points_f(DynModIPtr dyn_mod, double dt, const Mat_a2ap1& sigma_points, const Vec_u& u = Vec_u::Zero()) const {
        Eigen::Matrix<double, N_DIM_x, 2*N_DIM_a + 1> sigma_x_pred;
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            Vec_x x_i = sigma_points.template block<N_DIM_x, 1>(0, i);
            Vec_v v_i = sigma_points.template block<N_DIM_v, 1>(N_DIM_x, i);
            sigma_x_pred.col(i) = dyn_mod->f_d(dt, x_i, u, v_i);
        }
        return sigma_x_pred;
    }

    /** Propagate sigma points through h
     * @param sens_mod Sensor model
     * @param sigma_points Mat_a2ap1 Sigma points
     * @return Mat_z2ap1 sigma_z_pred
     */
    Mat_z2ap1 propagate_sigma_points_h(SensModIPtr sens_mod, const Mat_a2ap1& sigma_points) const {
        Mat_z2ap1 sigma_z_pred;
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            Vec_x x_i = sigma_points.template block<N_DIM_x, 1>(0, i);
            Vec_w w_i = sigma_points.template block<N_DIM_w, 1>(N_DIM_x + N_DIM_v, i);
            sigma_z_pred.col(i) = sens_mod->h(x_i, w_i);
        }
        return sigma_z_pred;
    }

    /** Estimate gaussian from sigma points
     * @param sigma_points Mat_n2ap1 Sigma points
     * @tparam n_dims Dimension of the gaussian
     * @return prob::MultiVarGauss<n_dims> 
     * @note This function is templated to allow for different dimensions of the gaussian
     */
    template<int n_dims>
    prob::MultiVarGauss<n_dims> estimate_gaussian(const Eigen::Matrix<double, n_dims, 2*N_DIM_a + 1>& sigma_points) const {
        // Predicted State Estimate x_k-
        Eigen::Vector<double, n_dims> mean = Eigen::Vector<double, n_dims>::Zero();
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            mean += W_x_[i] * sigma_points.col(i);
        }
        Eigen::Matrix<double, n_dims, n_dims> cov = Eigen::Matrix<double, n_dims, n_dims>::Zero();
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            cov += W_c_[i] * (sigma_points.col(i) - mean) * (sigma_points.col(i) - mean).transpose();
        }
        return {mean, cov};
    }
    
public:
    /** Perform one UKF prediction step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @param u Vec_u Control input
     * @return std::pair<Gauss_x, Gauss_z> Predicted state estimate, predicted measurement estimate
     */
	std::pair<Gauss_x, Gauss_z> predict(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x& x_est_prev, const Vec_u& u = Vec_u::Zero()) const override
    {
        Mat_a2ap1 sigma_points = get_sigma_points(dyn_mod, sens_mod, dt, x_est_prev);

        // Propagate sigma points through f and h
        Mat_x2ap1 sigma_x_pred = propagate_sigma_points_f(dyn_mod, dt, sigma_points, u);
        Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sens_mod, sigma_points);

        // Predicted State and Measurement Estimate x_k- and z_k-
        Gauss_x x_pred = estimate_gaussian<N_DIM_x>(sigma_x_pred);
        Gauss_z z_pred = estimate_gaussian<N_DIM_z>(sigma_z_pred);

        return {x_pred, z_pred};
    }

    /** Perform one UKF update step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param x_est_pred Predicted state estimate
     * @param z_est_pred Predicted measurement estimate
     * @param z_meas Measurement
     * @return Gauss_x Updated state estimate
     * @note Sigma points are generated from the predicted state estimate instead of the previous state estimate as is done in the 'step' method.
     */
    Gauss_x update(DynModIPtr dyn_mod, SensModIPtr sens_mod, const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) const override
    {
        // Generate sigma points from the predicted state estimate
        Mat_a2ap1 sigma_points = get_sigma_points(dyn_m, sensor_model_, 0.0, x_est_pred);

        // Extract the sigma points for the state
        Mat_x2ap1 sigma_x_pred = sigma_points.template block<N_DIM_x, 2*N_DIM_a + 1>(0, 0);

        // Predict measurement
        Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sens_mod, sigma_points);

        // Calculate cross-covariance
        Mat_xz P_xz = Mat_xz::Zero();
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            P_xz += W_c_[i] * (sigma_x_pred.col(i) - x_est_pred.mean()) * (sigma_z_pred.col(i) - z_est_pred.mean()).transpose();
        }

        // Calculate Kalman gain
        Mat_zz P_zz = z_est_pred.cov();
        Mat_xz K = P_xz * P_zz.llt().solve(Mat_zz::Identity());

        // Update state estimate
        Vec_x x_upd_mean = x_est_pred.mean() + K * (z_meas - z_est_pred.mean());
        Mat_xx x_upd_cov = x_est_pred.cov() - K * P_zz * K.transpose();
        Gauss_x x_est_upd = {x_upd_mean, x_upd_cov};

        return x_est_upd;
        
    }

    /** Perform one UKF prediction and update step
     * @param dyn_mod Dynamic model
     * @param sens_mod Sensor model
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @param z_meas Measurement
     * @param u Vec_u Control input
     * @return std::tuple<Gauss_x, Gauss_x, Gauss_z> Updated state estimate, predicted state estimate, predicted measurement estimate
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x& x_est_prev, const Vec_z& z_meas, const Vec_u& u) const override
    {
        Mat_a2ap1 sigma_points = get_sigma_points(dyn_mod, sens_mod, dt, x_est_prev);

        // Propagate sigma points through f and h
        Mat_x2ap1 sigma_x_pred = propagate_sigma_points_f(dyn_mod, dt, sigma_points, u);
        Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sens_mod, sigma_points);

        // Predicted State and Measurement Estimate x_k- and z_k-
        Gauss_x x_pred = estimate_gaussian<N_DIM_x>(sigma_x_pred);
        Gauss_z z_pred = estimate_gaussian<N_DIM_z>(sigma_z_pred);

        // Calculate cross-covariance
        Mat_xz P_xz = Mat_xz::Zero();
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            P_xz += W_c_[i] * (sigma_x_pred.col(i) - x_pred.mean()) * (sigma_z_pred.col(i) - z_pred.mean()).transpose();
        }

        // Calculate Kalman gain
        Mat_zz P_zz = z_pred.cov();
        Mat_xz K = P_xz * P_zz.llt().solve(Mat_zz::Identity());

        // Update state estimate
        Vec_x x_upd_mean = x_pred.mean() + K * (z_meas - z_pred.mean());
        Mat_xx x_upd_cov = x_pred.cov() - K * P_zz * K.transpose();
        Gauss_x x_est_upd = {x_upd_mean, x_upd_cov};

        return {x_est_upd, x_pred, z_pred};
    }

    /** Perform one UKF prediction step
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @param u Vec_u Control input
     * @return std::pair<Gauss_x, Gauss_z> Predicted state estimate, predicted measurement estimate
     */
    std::pair<Gauss_x, Gauss_z> predict(double dt, const Gauss_x& x_est_prev, const Vec_u& u = Vec_u::Zero()) const {
        // check if dynamic_model_ and sensor_model_ are set
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("UKF::predict() called without dynamic_model_ or sensor_model_ set.");
        }
        return predict(dynamic_model_, sensor_model_, dt, x_est_prev, u);
    }

    /** Perform one UKF update step
     * @param x_est_pred Predicted state estimate
     * @param z_est_pred Predicted measurement estimate
     * @param z_meas Measurement
     * @return Gauss_x Updated state estimate
     */
    Gauss_x update(const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) const {
        // check if dynamic_model_ and sensor_model_ are set
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("UKF::update() called without dynamic_model_ or sensor_model_ set.");
        }
        return update(dynamic_model_, sensor_model_, x_est_pred, z_est_pred, z_meas);
    }

    /** Perform one UKF prediction and update step
     * @param dt Time step
     * @param x_est_prev Previous state estimate
     * @param z_meas Measurement
     * @param u Vec_u Control input
     * @return std::tuple<Gauss_x, Gauss_x, Gauss_z> Updated state estimate, predicted state estimate, predicted measurement estimate
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(double dt, const Gauss_x& x_est_prev, const Vec_z& z_meas, const Vec_u& u = Vec_u::Zero()) const {
        // check if dynamic_model_ and sensor_model_ are set
        if (!dynamic_model_ || !sensor_model_) {
            throw std::runtime_error("UKF::step() called without dynamic_model_ or sensor_model_ set.");
        }
        return step(dynamic_model_, sensor_model_, dt, x_est_prev, z_meas, u);
    }

private:
    const DynModIPtr dynamic_model_;
    const SensModIPtr sensor_model_;

    // Parameters for UKF
    double ALPHA_;
    double BETA_; 
    double KAPPA_; 
    double LAMBDA_; // Parameter for spread of sigma points
    double GAMMA_; // Scaling factor for spread of sigma points
    std::vector<double> W_x_, W_c_; // Weighting factors for sigma points

};

/** Unscented Kalman Filter with dimensions defined by the dynamic model and sensor model.
 * @tparam DynModT Dynamic model type
 * @tparam SensModT Sensor model type
 */
template <typename DynModT, typename SensModT>
using UKF_M = UKF<DynModT::N_DIM_x, SensModT::N_DIM_z, DynModT::N_DIM_u, DynModT::N_DIM_v, SensModT::N_DIM_w>;

}  // namespace filter
}  // namespace vortex