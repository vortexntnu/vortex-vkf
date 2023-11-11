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
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/sensor_model.hpp>

namespace vortex {
namespace filters {

template<typename DynamicModelT, typename SensorModelT>
class UKF {
public:
    static constexpr int N_DIM_x = DynamicModelT::N_DIM_x;
    static constexpr int N_DIM_u = DynamicModelT::N_DIM_u;
    static constexpr int N_DIM_z = SensorModelT::N_DIM_z;
    static constexpr int N_DIM_v = DynamicModelT::N_DIM_v;
    static constexpr int N_DIM_w = SensorModelT::N_DIM_w;

    using DynModI  = models::DynamicModelBaseI<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModI = models::SensorModelBaseI<N_DIM_x, N_DIM_z, N_DIM_w>;

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

    // Augmented state dimension
    static constexpr int N_DIM_a = N_DIM_x + N_DIM_v + N_DIM_w;
    using Vec_a     = Eigen::Vector<double, N_DIM_a>;
    using Mat_aa    = Eigen::Matrix<double, N_DIM_a, N_DIM_a>;
    using Mat_x2ap1 = Eigen::Matrix<double, N_DIM_x, 2*N_DIM_a + 1>;
    using Mat_z2ap1 = Eigen::Matrix<double, N_DIM_z, 2*N_DIM_a + 1>;
    using Mat_a2ap1 = Eigen::Matrix<double, N_DIM_a, 2*N_DIM_a + 1>;

    /** Unscented Kalman Filter
     * @param dynamic_model Dynamic model
     * @param sensor_model Sensor model
     * @tparam DynamicModelT Dynamic model type. Can take any model that implements f_d and Q_d
     * @tparam SensorModelT Sensor model type. Can take any model that implements h and R
     */
    UKF(std::shared_ptr<DynModI> dynamic_model, std::shared_ptr<SensModI> sensor_model) : UKF(dynamic_model, sensor_model, 1.0, 2.0, 0.0) {}
        
    /** Unscented Kalman Filter
     * @param dynamic_model Dynamic model
     * @param sensor_model Sensor model
     */
    UKF(std::shared_ptr<DynModI> dynamic_model, std::shared_ptr<SensModI> sensor_model, double alpha, double beta, double kappa)
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
     * @param x_est Gauss_x State estimate
     * @return Mat_a2ap1 sigma_points
    */
    Mat_a2ap1 get_sigma_points(const Gauss_x& x_est, double dt) 
    {
        Mat_xx P = x_est.cov();
        Mat_vv Q = dynamic_model_->Q_d(x_est.mean(), dt);
        Mat_ww R = sensor_model_->R(x_est.mean());
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

        // Calculate sigma points
        Mat_a2ap1 sigma_points;

        // Use the symmetric sigma point set
        sigma_points.col(0) = x_a;
        for (int i = 1; i <= N_DIM_a; i++) {
            sigma_points.col(i)       = x_a + GAMMA_ * sqrt_P_a.col(i - 1);
            sigma_points.col(i + N_DIM_a) = x_a - GAMMA_ * sqrt_P_a.col(i - 1);
        }
        return sigma_points;
    }

    /** Propagate sigma points through f
     * @param sigma_points Mat_a2ap1 Sigma points
     * @param u Vec_u Control input
     * @param dt Time step
     * @return Mat_x2ap1 sigma_x_pred
     */
    Mat_x2ap1 propagate_sigma_points_f(const Mat_a2ap1& sigma_points, const Vec_u& u, double dt) {
        Eigen::Matrix<double, N_DIM_x, 2*N_DIM_a + 1> sigma_x_pred;
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            Vec_x x_i = sigma_points.template block<N_DIM_x, 1>(0, i);
            Vec_v v_i = sigma_points.template block<N_DIM_v, 1>(N_DIM_x, i);
            sigma_x_pred.col(i) = dynamic_model_->f_d(x_i, u, v_i, dt);
        }
        return sigma_x_pred;
    }

    /** Propagate sigma points through h
     * @param sigma_points Mat_a2ap1 Sigma points
     * @return Mat_z2ap1 sigma_z_pred
     */
    Mat_z2ap1 propagate_sigma_points_h(const Mat_a2ap1& sigma_points) {
        Mat_z2ap1 sigma_z_pred;
        for (int i = 0; i < 2*N_DIM_a + 1; i++) {
            Vec_x x_i = sigma_points.template block<N_DIM_x, 1>(0, i);
            Vec_w w_i = sigma_points.template block<N_DIM_w, 1>(N_DIM_x + N_DIM_v, i);
            sigma_z_pred.col(i) = sensor_model_->h(x_i, w_i);
        }
        return sigma_z_pred;
    }

    /** Estimate gaussian from sigma points
     * @param sigma_points Mat_n2ap1 Sigma points
     * @tparam n_dims Dimension of the gaussian
     * @return prob::MultiVarGauss<n_dims> 
     */
    template<int n_dims>
    prob::MultiVarGauss<n_dims> estimate_gaussian(const Eigen::Matrix<double, n_dims, 2*N_DIM_a + 1>& sigma_points) {
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
     * @param x_est_prev Previous state estimate
     * @param u Vec_u Control input
     * @param dt Time step
     */
    std::pair<Gauss_x, Gauss_z> predict(const Gauss_x& x_est_prev, Vec_u u, double dt) 
    {
        Mat_a2ap1 sigma_points = get_sigma_points(x_est_prev, dt);

        // Propagate sigma points through f and h
        Mat_x2ap1 sigma_x_pred = propagate_sigma_points_f(sigma_points, u, dt);
        Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sigma_points);

        // Predicted State and Measurement Estimate x_k- and z_k-
        Gauss_x x_pred = estimate_gaussian<N_DIM_x>(sigma_x_pred);
        Gauss_z z_pred = estimate_gaussian<N_DIM_z>(sigma_z_pred);

        return {x_pred, z_pred};
    }

    /** Perform one UKF prediction and update step
     * @param x_est_prev Previous state estimate
     * @param z_meas Measurement
     * @param u Vec_u Control input
     * @param dt Time step
     */
    std::tuple<Gauss_x, Gauss_x, Gauss_z> step(const Gauss_x& x_est_prev, const Vec_z& z_meas, Vec_u u, double dt)
    {
        Mat_a2ap1 sigma_points = get_sigma_points(x_est_prev, dt);

        // Propagate sigma points through f and h
        Mat_x2ap1 sigma_x_pred = propagate_sigma_points_f(sigma_points, u, dt);
        Mat_z2ap1 sigma_z_pred = propagate_sigma_points_h(sigma_points);

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

private:
    std::shared_ptr<DynModI> dynamic_model_;
    std::shared_ptr<SensModI> sensor_model_;

    // Parameters for UKF
    double ALPHA_, BETA_, KAPPA_, LAMBDA_, GAMMA_;
    // Scaling factors
    std::vector<double> W_x_, W_c_;

};

}  // namespace filters
}  // namespace vortex