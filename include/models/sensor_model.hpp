/**
 * @file sensor_model.hpp
 * @author Eirik Kolås
 * @brief Sensor model interface. Based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-10-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <probability/multi_var_gauss.hpp>

namespace vortex {
namespace models {


template <int n_dim_x, int n_dim_z>
/**
 * @brief Interface for sensor models.
 * 
 */
class SensorModelI {
public:
    static constexpr int N_DIM_x = n_dim_x; // Declare so that children of this class can reference it
    static constexpr int N_DIM_z = n_dim_z; // Declare so that children of this class can reference it
    using Vec_z  = Eigen::Vector<double, N_DIM_z>;
    using Vec_x  = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Mat_zx = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_xz = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
    using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_z = prob::MultiVarGauss<N_DIM_z>;

    virtual ~SensorModelI() = default;
    /**
     * @brief Sensor Model
     * @param x State
     * @return Vec_z
     */
    virtual Vec_z h(const Vec_x& x) const = 0;
    /**
     * @brief Jacobian of sensor model with respect to state
     * @param x State
     * @return Mat_zx 
     */
    virtual Mat_zx H(const Vec_x& x) const = 0;
    /**
     * @brief Noise covariance matrix
     * @param x State
     * @return Mat_zz 
     */
    virtual Mat_zz R(const Vec_x& x) const = 0;

    /**
     * @brief Get the predicted measurement distribution given a state estimate. Updates the covariance
     * 
     * @param x_est Vec_x estimate
     * @return prob::MultiVarGauss 
     */
    virtual Gauss_z pred_from_est(const Gauss_x& x_est) const 
    {
        Mat_xx P = x_est.cov();
        Mat_zx H = this->H(x_est.mean());
        Mat_zz R = this->R(x_est.mean());

        return {this->h(x_est.mean()), H * P * H.transpose() + R};
    }

    /**
     * @brief Get the predicted measurement distribution given a state. Does not update the covariance
     * @param x Vec_x
     * @return prob::MultiVarGauss 
     */
    virtual Gauss_z pred_from_state(const Vec_x& x) const 
    {
        return {this->h(x), this->R(x)};
    }
};

}  // namespace models
}  // namespace vortex