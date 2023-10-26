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

#include <Eigen/Dense>
#include <probability/multi_var_gauss.hpp>

namespace vortex {
namespace filters {

template <int n_dim>
class DynamicModel {
    using State = Eigen::Matrix<double, n_dim, 1>;
    using Mat_xx = Eigen::Matrix<double, n_dim, n_dim>;
public:

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
        return Eigen::expm(A_c(x) * dt);
    }

    /** Discrete time process noise
     * @param x State
     * @param dt Time step
     * @return Matrix Process noise covariance
     */
    virtual Mat_xx Q_d(const State& x, double dt)
    {
        // See https://en.wikipedia.org/wiki/Discretization#Discretization_of_process_noise for more info
        Mat_xx A_c = A_c(x);
        Mat_xx Q_c = Q_c(x);

        Mat_xx F;
        // clang-format off
        F << -A_c           , Q_c,
             Mat_xx::Zeros(), A_c.transpose();
        // clang-format on
        Eigen::Matrix<double, 2 * n_dim, 2 * n_dim> G = Eigen::expm(F * dt);
        return G.template block<n_dim, n_dim>(0, n_dim) * G.template block<n_dim, n_dim>(n_dim, n_dim).transpose();
    }
    

    /** Propagate state estimate through dynamics
     * @param x_est State estimate
     * @param dt Time step
     * @return State
     */
    virtual prob::MultiVarGauss<n_dim> pred_from_est(const prob::MultiVarGauss<n_dim>& x_est, double dt)
    {
        Mat_xx P = x_est.cov();
        Mat_xx F_d = F_d(x_est.mean(), dt);
        Mat_xx Q_d = Q_d(x_est.mean(), dt);
        prob::MultiVarGauss x_est_pred(f_d(x_est.mean(), dt), F_d * P * F_d.transpose() + Q_d);

        return x_est_pred;
    }

};

}  // namespace filters
}  // namespace vortex
