/**
 * @file dynamic_model.hpp
 * @author Eirik Kolås
 * @brief Dynamic model interface. Based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-10-26
 */
#pragma once
#include <Eigen/Dense>
#include <random>
#include <eigen3/unsupported/Eigen/MatrixFunctions> // For exp
#include <vortex_filtering/probability/multi_var_gauss.hpp>


namespace vortex {
namespace models {

/**
 * @brief Interface for dynamic models with dynamic size dimensions.
 * The purpose of this class is to provide a common interface for dynamic models of any dimension so that they can be used in the same way.
 * This class is not meant to be inherited from. Use DynamicModelI instead. 
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 */
class DynamicModelX {
public:
    // Using dynamic Eigen types
    using VecX = Eigen::VectorXd;
    using MatXX = Eigen::MatrixXd;
    using GaussX = prob::MultiVarGauss<Eigen::Dynamic>;

    // Constructor to initialize the dimensions
    DynamicModelX(int dim_x, int dim_u, int dim_v)
        : dim_x_(dim_x), dim_u_(dim_u), dim_v_(dim_v) {}

    virtual ~DynamicModelX() = default;

    // Discrete time dynamics (pure virtual function)
    virtual VecX f_dX(const VecX& x, const VecX& u, const VecX& v, double dt) const = 0;

    // Discrete time process noise (pure virtual function)
    virtual MatXX Q_dX(const VecX& x, double dt) const = 0;

    // Sample from the discrete time dynamics
    VecX sample_f_dX(const VecX& x, const VecX& u, double dt, std::mt19937& gen) const {
        GaussX v = {VecX::Zero(dim_v_), Q_dX(x, dt)};
        return f_dX(x, u, v.sample(gen), dt);
    }

    // Sample from the discrete time dynamics
    VecX sample_f_dX(const VecX& x, double dt) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        return sample_f_dX(x, VecX::Zero(dim_u_), dt, gen);
    }

    int get_dim_x() const { return dim_x_; }
    int get_dim_u() const { return dim_u_; }
    int get_dim_v() const { return dim_v_; }

protected:
    const int dim_x_;  // State dimension
    const int dim_u_;  // Input dimension
    const int dim_v_;  // Process noise dimension
};




/**
 * @brief Interface for dynamic models with static size dimensions
 * 
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 */
template <int n_dim_x, int n_dim_u, int n_dim_v>
class DynamicModelI : public DynamicModelX {
public:
    using BaseX = DynamicModelX;
    static constexpr int N_DIM_x = n_dim_x; // Declare so that children of this class can reference it
    using Vec_x = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;

    static constexpr int N_DIM_u = n_dim_u; // Declare so that children of this class can reference it
    using Vec_u = Eigen::Vector<double, N_DIM_u>;
    using Mat_uu = Eigen::Matrix<double, N_DIM_u, N_DIM_u>;
    using Mat_xu = Eigen::Matrix<double, N_DIM_x, N_DIM_u>;

    static constexpr int N_DIM_v = n_dim_v; // Declare so that children of this class can reference it
    using Vec_v = Eigen::Vector<double, N_DIM_v>;
    using Mat_vv = Eigen::Matrix<double, N_DIM_v, N_DIM_v>;
    using Mat_xv = Eigen::Matrix<double, N_DIM_x, N_DIM_v>;
    using Gauss_v = prob::MultiVarGauss<N_DIM_v>;

    DynamicModelI() : DynamicModelX(N_DIM_x, N_DIM_u, N_DIM_v) {}
    virtual ~DynamicModelI() = default;

    /** Discrete time dynamics
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param v Vec_v Process noise
     * @param dt Time step
     * @return State_dot
     */
    virtual Vec_x f_d(const Vec_x& x, const Vec_u& u, const Vec_v& v, double dt) const = 0;

    /** Discrete time process noise
     * @param x Vec_x State
     */
    virtual Mat_vv Q_d(const Vec_x& x, double dt) const = 0;

    /** Sample from the discrete time dynamics
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param dt Time step
     * @param gen Random number generator (For deterministic behaviour)
     * @return Vec_x State
     */
    Vec_x sample_f_d(const Vec_x& x, const Vec_u& u, double dt, std::mt19937& gen) const
    {
        Gauss_v v = {Vec_v::Zero(), Q_d(x, dt)};
        return f_d(x, u, v.sample(gen), dt);
    }

    /** Sample from the discrete time dynamics
     * @param x Vec_x State
     * @param dt Time step
     * @return Vec_x State
     */
    Vec_x sample_f_d(const Vec_x& x, double dt) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        return sample_f_d(x, Vec_u::Zero(), dt, gen);
    }

    // Override dynamic size functions to use static size functions
protected:
    BaseX::VecX f_dX(const BaseX::VecX& x, const BaseX::VecX& u, const BaseX::VecX& v, double dt) const override
    {
        Vec_x x_fixed = x;
        Vec_u u_fixed = u;
        Vec_v v_fixed = v;
        return f_d(x_fixed, u_fixed, v_fixed, dt);
    }

    BaseX::MatXX Q_dX(const BaseX::VecX& x, double dt) const override
    {
        Vec_x x_fixed = x;
        return Q_d(x_fixed, dt);
    }

};


/**
 * @brief Interface for dynamic models. It assumes additive noise and no input u.
 *  This is suitable for the EKF as it has built in functions for excact discretization of a linear model provided an A_c and Q_c.
 * 
 * @tparam n_dim_x  State dimension
 */
template <int n_dim_x>
class DynamicModelEKFI : public DynamicModelI<n_dim_x, n_dim_x, n_dim_x> {
public:
    static constexpr int N_DIM_x = n_dim_x; // Declare so that children of this class can reference it
    static constexpr int N_DIM_u = n_dim_x; // Declare so that children of this class can reference it
    static constexpr int N_DIM_v = n_dim_x; // Declare so that children of this class can reference it

    using Vec_x = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;

    virtual ~DynamicModelEKFI() = default;

    /** Continuos time dynamics.
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

    /** Discrete time dynamics.
     * Overriding DynamicModelI::f_d
     * @param x Vec_x
     * @param u Vec_u
     * @param v Vec_v
     * @param dt Time step
     * @return Vec_x
     */
    Vec_x f_d(const Vec_x& x, const Vec_x&, const Vec_x& v, double dt) const override
    {
        return f_d(x, dt) + v;
    }
    /** Discrete time dynamics
     * @param x Vec_x
     * @param dt Time step
     * @return Vec_x
     */
    Vec_x f_d(const Vec_x& x, double dt) const
    {
        return F_d(x, dt) * x;
    }

    /** Jacobian of discrete time dynamics
     * @param x Vec_x
     * @param dt Time step
     * @return State_jac
     */
    Mat_xx F_d(const Vec_x& x, double dt) const
    {
        // Use (4.58) from the book
        return (A_c(x) * dt).exp();
    }

    /** Discrete time process noise.
     * Overriding DynamicModelI::Q_d
     * @param x Vec_x
     * @param dt Time step
     * @return Matrix Process noise covariance
     */
    Mat_xx Q_d(const Vec_x& x, double dt) const override
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
    Gauss_x pred_from_est(const Gauss_x& x_est, double dt) const
    {
        Mat_xx P = x_est.cov();
        Mat_xx F_d = this->F_d(x_est.mean(), dt);
        Mat_xx Q_d = this->Q_d(x_est.mean(), dt);
        Gauss_x x_est_pred(f_d(x_est.mean(), dt), F_d * P * F_d.transpose() + Q_d);

        return x_est_pred;
    }

    /** Get the predicted state distribution given a state
     * @param x Vec_x
     * @param dt Time step
     * @return Vec_x
     */
    Gauss_x pred_from_state(const Vec_x& x, double dt) const
    {
        Mat_xx Q_d = this->Q_d(x, dt);
        Gauss_x x_est_pred(this->f_d(x, dt), Q_d);

        return x_est_pred;
    }

    // Give access to the base class functions
    using DynamicModelI<n_dim_x, n_dim_x, n_dim_x>::sample_f_d;

    /** Sample from the discrete time dynamics
     * @param x Vec_x State
     * @param dt Time step
     * @return Vec_x State
     */
    Vec_x sample_f_d(const Vec_x& x, double dt) const
    {
        return sample_f_d(x, Vec_x::Zero(), dt);
    }

    /** Sample from the discrete time dynamics
     * @param x Vec_x State
     * @param dt Time step
     * @param gen Random number generator (For deterministic behaviour)
     * @return Vec_x State
     */
    Vec_x sample_f_d(const Vec_x& x, double dt, std::mt19937& gen) const
    {
        return sample_f_d(x, Vec_x::Zero(), dt, gen);
    }

};

}  // namespace models
}  // namespace vortex
