/**
 * @file dynamic_model.hpp
 * @author Eirik Kolås
 * @brief Dynamic model interface. Based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-10-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions> // For exp
#include <probability/multi_var_gauss.hpp>
#include <integration_methods/ERK_methods.hpp>

namespace vortex {
namespace models {

template <int n_dim_x>
class DynamicModel {
public:
    static constexpr int N_DIM_x = n_dim_x;
    using State = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    virtual ~DynamicModel() = default;

    /** Continuos time dynamics
     * @param x State
     * @return State_dot
     */
    virtual State f_c(const State& x) const = 0;
    
    /** Jacobian of continuous time dynamics
     * @param x State
     * @return State_jac
     */
    virtual Mat_xx A_c(const State& x) const = 0; 

    /** Continuous time process noise
     * @param x State
     * @return Matrix Process noise covariance
     */
    virtual Mat_xx Q_c(const State& x) const = 0;

    /** Discrete time dynamics
     * @param x State
     * @param dt Time step
     * @return State
     */
    virtual State f_d(const State& x, double dt) 
    {
        return F_d(x, dt) * x;
    }

    /** Jacobian of discrete time dynamics
     * @param x State
     * @param dt Time step
     * @return State_jac
     */
    virtual Mat_xx F_d(const State& x, double dt) 
    {
        // Use (4.58) from the book
        return (A_c(x) * dt).exp();
    }

    /** Discrete time process noise
     * @param x State
     * @param dt Time step
     * @return Matrix Process noise covariance
     */
    virtual Mat_xx Q_d(const State& x, double dt)
    {
        // See https://en.wikipedia.org/wiki/Discretization#Discretization_of_process_noise for more info

        Mat_xx A_c = this->A_c(x);
        Mat_xx Q_c = this->Q_c(x);

        Eigen::Matrix<double, 2 * N_DIM_x, 2 * N_DIM_x> F;
        // clang-format off
        F << -A_c          , Q_c,
             Mat_xx::Zero(), A_c.transpose();
        // clang-format on
        Eigen::Matrix<double, 2 * N_DIM_x, 2 * N_DIM_x> G = (F * dt).exp();
        return G.template block<N_DIM_x, N_DIM_x>(0, N_DIM_x) * G.template block<N_DIM_x, N_DIM_x>(N_DIM_x, N_DIM_x).transpose();
    }
    

    /** Get the predicted state distribution given a state estimate
     * @param x_est State estimate
     * @param dt Time step
     * @return State
     */
    virtual prob::MultiVarGauss<N_DIM_x> pred_from_est(const prob::MultiVarGauss<N_DIM_x>& x_est, double dt)
    {
        Mat_xx P = x_est.cov();
        Mat_xx F_d = this->F_d(x_est.mean(), dt);
        Mat_xx Q_d = this->Q_d(x_est.mean(), dt);
        prob::MultiVarGauss<N_DIM_x> x_est_pred(this->f_d(x_est.mean(), dt), F_d * P * F_d.transpose() + Q_d);

        return x_est_pred;
    }

    /** Get the predicted state distribution given a state
     * @param x State
     * @param dt Time step
     * @return State
     */
    virtual prob::MultiVarGauss<N_DIM_x> pred_from_state(const State& x, double dt)
    {
        Mat_xx Q_d = this->Q_d(x, dt);
        prob::MultiVarGauss<N_DIM_x> x_est_pred(this->f_d(x, dt), Q_d);

        return x_est_pred;
    }

};

}  // namespace models
}  // namespace vortex
