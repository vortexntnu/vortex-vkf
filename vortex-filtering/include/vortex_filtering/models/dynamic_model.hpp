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
#include <vortex_filtering/probability/multi_var_gauss.hpp>


namespace vortex {
namespace models {

/**
 * @brief Interface for dynamic models. 
 * 
 */
template <int n_dim_x>
class DynamicModelI {
public:
    static constexpr int N_DIM_x = n_dim_x; // Declare so that children of this class can reference it
    using Vec_x = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;

    virtual ~DynamicModelI() = default;

    /** Continuos time dynamics
     * @param x Vec_x
     * @return State_dot
     */
    virtual Vec_x f_c(const Vec_x& x) const = 0;
    
    /** Jacobian of continuous time dynamics
     * @param x Vec_x
     * @return State_jac
     */
    virtual Mat_xx A_c(const Vec_x& x) const = 0; 

    /** Continuous time process noise
     * @param x Vec_x
     * @return Matrix Process noise covariance
     */
    virtual Mat_xx Q_c(const Vec_x& x) const = 0;

    /** Discrete time dynamics
     * @param x Vec_x
     * @param dt Time step
     * @return Vec_x
     */
    virtual Vec_x f_d(const Vec_x& x, double dt) const
    {
        return F_d(x, dt) * x;
    }

    /** Jacobian of discrete time dynamics
     * @param x Vec_x
     * @param dt Time step
     * @return State_jac
     */
    virtual Mat_xx F_d(const Vec_x& x, double dt) const
    {
        // Use (4.58) from the book
        return (A_c(x) * dt).exp();
    }

    /** Discrete time process noise
     * @param x Vec_x
     * @param dt Time step
     * @return Matrix Process noise covariance
     */
    virtual Mat_xx Q_d(const Vec_x& x, double dt) const
    {
        // See https://en.wikipedia.org/wiki/Discretization#Discretization_of_process_noise for more info

        Mat_xx A_c = this->A_c(x);
        Mat_xx Q_c = this->Q_c(x);

        Eigen::Matrix<double, 2 * N_DIM_x, 2 * N_DIM_x> v_1;
        v_1 << -A_c, Q_c, Mat_xx::Zero(), A_c.transpose();
        v_1 *= dt;
        v_1 = v_1.exp();
        Mat_xx F_d = v_1.template block<N_DIM_x, N_DIM_x>(N_DIM_x, N_DIM_x).transpose();
        Mat_xx F_d_inv_Q_d = v_1.template block<N_DIM_x, N_DIM_x>(0, N_DIM_x);
        Mat_xx Q_d = F_d * F_d_inv_Q_d;

        return Q_d;
    }
    

    /** Get the predicted state distribution given a state estimate
     * @param x_est Vec_x estimate
     * @param dt Time step
     * @return Vec_x
     */
    virtual Gauss_x pred_from_est(const Gauss_x& x_est, double dt) const
    {
        Mat_xx P = x_est.cov();
        Mat_xx F_d = this->F_d(x_est.mean(), dt);
        Mat_xx Q_d = this->Q_d(x_est.mean(), dt);
        Gauss_x x_est_pred(this->f_d(x_est.mean(), dt), F_d * P * F_d.transpose() + Q_d);

        return x_est_pred;
    }

    /** Get the predicted state distribution given a state
     * @param x Vec_x
     * @param dt Time step
     * @return Vec_x
     */
    virtual Gauss_x pred_from_state(const Vec_x& x, double dt) const
    {
        Mat_xx Q_d = this->Q_d(x, dt);
        Gauss_x x_est_pred(this->f_d(x, dt), Q_d);

        return x_est_pred;
    }

};

}  // namespace models
}  // namespace vortex
