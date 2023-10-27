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


template <int N_DIM_x, int N_DIM_z>
class SensorModel {
public:
    using Measurement = Eigen::Vector<double, N_DIM_z>;
    using State       = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx      = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Mat_zx      = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_zz      = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;

    virtual ~SensorModel() = default;

    virtual Measurement h(const State& x) const = 0;
    virtual Mat_zx H(const State& x) const = 0;
    virtual Mat_zz R(const State& x) const = 0;

    /**
     * @brief Get the predicted measurement distribution given a state estimate. Updates the covariance
     * 
     * @param x_est State estimate
     * @return prob::MultiVarGauss 
     */
    virtual prob::MultiVarGauss<N_DIM_z> pred_from_est(const prob::MultiVarGauss<N_DIM_x>& x_est) const 
    {
        Mat_xx P = x_est.cov();
        Mat_zx H = this->H(x_est.mean());
        Mat_zz R = this->R(x_est.mean());

        return {this->h(x_est.mean()), H * P * H.transpose() + R};
    }

    /**
     * @brief Get the predicted measurement distribution given a state. Does not update the covariance
     * @param x State
     * @return prob::MultiVarGauss 
     */
    virtual prob::MultiVarGauss<N_DIM_z> pred_from_state(const State& x) const 
    {
        return {this->h(x), this->R(x)};
    }
};

}  // namespace models
}  // namespace vortex