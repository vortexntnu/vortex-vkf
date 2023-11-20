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
#include <functional>
#include <eigen3/unsupported/Eigen/MatrixFunctions> // For exp
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/numerical_integration/erk_methods.hpp>


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
    virtual VecX f_dX(double dt, const VecX& x, const VecX& u, const VecX& v) const = 0;

    // Discrete time process noise (pure virtual function)
    virtual MatXX Q_dX(double dt, const VecX& x) const = 0;

    // Sample from the discrete time dynamics
    virtual VecX sample_f_dX(double dt, const VecX& x, const VecX& u, std::mt19937& gen) const = 0;

    // Sample from the discrete time dynamics
    virtual VecX sample_f_dX(double dt, const VecX& x) const = 0;

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
    static constexpr int N_DIM_u = n_dim_u; // Declare so that children of this class can reference it
    static constexpr int N_DIM_v = n_dim_v; // Declare so that children of this class can reference it

    using Vec_x = Eigen::Vector<double, N_DIM_x>;
    using Vec_u = Eigen::Vector<double, N_DIM_u>;
    using Vec_v = Eigen::Vector<double, N_DIM_v>;

    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Mat_xu = Eigen::Matrix<double, N_DIM_x, N_DIM_u>;
    using Mat_xv = Eigen::Matrix<double, N_DIM_x, N_DIM_v>;

    using Mat_uu = Eigen::Matrix<double, N_DIM_u, N_DIM_u>;
    using Mat_vv = Eigen::Matrix<double, N_DIM_v, N_DIM_v>;

    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_v = prob::MultiVarGauss<N_DIM_v>;

    DynamicModelI() : DynamicModelX(N_DIM_x, N_DIM_u, N_DIM_v) {}
    virtual ~DynamicModelI() = default;

    /** Discrete time dynamics
     * @param dt Time step
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param v Vec_v Process noise
     * @return Vec_x Next state
     */
    virtual Vec_x f_d(double dt, const Vec_x& x, const Vec_u& u, const Vec_v& v) const = 0;

    /** Discrete time process noise covariance matrix
     * @param dt Time step
     * @param x Vec_x State
     */
    virtual Mat_vv Q_d(double dt, const Vec_x& x) const = 0;

    /** Sample from the discrete time dynamics
     * @param dt Time step
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param gen Random number generator (For deterministic behaviour)
     * @return Vec_x Next state
     */
    Vec_x sample_f_d(double dt, const Vec_x& x, const Vec_u& u, std::mt19937& gen) const
    {
        Gauss_v v = {Vec_v::Zero(), Q_d(dt, x)};
        return f_d(dt, x, u, v.sample(gen));
    }

    /** Sample from the discrete time dynamics
     * @param dt Time step
     * @param x Vec_x State
     * @return Vec_x Next state
     */
    Vec_x sample_f_d(double dt, const Vec_x& x) const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        return sample_f_d(dt, x, Vec_u::Zero(), gen);
    }

protected:
    // Override dynamic size functions to use static size functions
    BaseX::VecX f_dX(double dt, const BaseX::VecX& x, const BaseX::VecX& u, const BaseX::VecX& v) const override
    {
        Vec_x x_fixed = x;
        Vec_u u_fixed = u;
        Vec_v v_fixed = v;
        return f_d(dt, x_fixed, u_fixed, v_fixed);
    }

    BaseX::MatXX Q_dX(double dt, const BaseX::VecX& x) const override
    {
        Vec_x x_fixed = x;
        return Q_d(dt, x_fixed);
    }

    BaseX::VecX sample_f_dX(double dt, const BaseX::VecX& x, const BaseX::VecX& u, std::mt19937& gen) const override
    {
        Vec_x x_fixed = x;
        Vec_u u_fixed = u;
        return sample_f_d(dt, x_fixed, u_fixed, gen);
    }

    BaseX::VecX sample_f_dX(double dt, const BaseX::VecX& x) const override
    {
        Vec_x x_fixed = x;
        return sample_f_d(dt, x_fixed);
    }

};



/**
 * @brief Continuous Time Dynamic Model Interface.
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension (Default: n_dim_x)
 * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
 */
template <int n_dim_x, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x>
class DynamicModelCT : virtual public DynamicModelI<n_dim_x, n_dim_u, n_dim_v> {
public:
    using BaseI = DynamicModelI<n_dim_x, n_dim_u, n_dim_v>;
    static constexpr int N_DIM_x = n_dim_x;
    static constexpr int N_DIM_u = n_dim_u;
    static constexpr int N_DIM_v = n_dim_v;

    using Vec_x = typename BaseI::Vec_x;
    using Vec_u = typename BaseI::Vec_u;
    using Vec_v = typename BaseI::Vec_v;

    using Mat_xx = typename BaseI::Mat_xx;
    using Mat_xu = typename BaseI::Mat_xu;
    using Mat_xv = typename BaseI::Mat_xv;

    using Mat_uu = typename BaseI::Mat_uu;
    using Mat_vv = typename BaseI::Mat_vv;

    using Dyn_mod_func = std::function<Vec_x(double t, const Vec_x& x)>;

    /** Continuous Time Dynamic Model Interface
     * @tparam n_dim_x  State dimension
     * @tparam n_dim_u  Input dimension
     * @tparam n_dim_v  Process noise dimension
     */
    DynamicModelCT() : BaseI() {}
    virtual ~DynamicModelCT() = default;

    /** Continuous time dynamics
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param v Vec_v Process noise
     * @return Vec_x State_dot
     */
    virtual Vec_x f_c(const Vec_x& x, const Vec_u& u, const Vec_v& v) const = 0;

    /** Discrete time process noise covariance matrix
     * @param dt Time step
     * @param x Vec_x State
     */
    virtual Mat_vv Q_d(double dt, const Vec_x& x) const override = 0;

protected:
    // Discrete time stuff

    /** Discrete time dynamics. Uses RK4 integration. Assumes constant input and process noise during the time step.
     * Overriding DynamicModelI::f_d
     * @param dt Time step
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param v Vec_v Process noise
     * @return Vec_x Next state
     */
    virtual Vec_x f_d(double dt, const Vec_x& x, const Vec_u& u, const Vec_v& v) const override
    {
        Dyn_mod_func f_c = [this, &u, &v](double, const Vec_x& x) { return this->f_c(x, u, v); };
        return vortex::integrator::RK4<N_DIM_x>::integrate(f_c, dt, x);
    }

};


/** Linear Time Variant Dynamic Model Interface.
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 */
template <int n_dim_x, int n_dim_u, int n_dim_v>
class DynamicModelLTV : virtual public DynamicModelI<n_dim_x, n_dim_u, n_dim_v> {
public:
    using BaseI = DynamicModelI<n_dim_x, n_dim_u, n_dim_v>;
    static constexpr int N_DIM_x = n_dim_x;
    static constexpr int N_DIM_u = n_dim_u;
    static constexpr int N_DIM_v = n_dim_v;

    using Vec_x = typename BaseI::Vec_x;
    using Vec_u = typename BaseI::Vec_u;
    using Vec_v = typename BaseI::Vec_v;

    using Mat_xx = typename BaseI::Mat_xx;
    using Mat_xu = typename BaseI::Mat_xu;
    using Mat_xv = typename BaseI::Mat_xv;

    using Mat_uu = typename BaseI::Mat_uu;
    using Mat_vv = typename BaseI::Mat_vv;

    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_v = prob::MultiVarGauss<N_DIM_v>;

    DynamicModelLTV() : BaseI() {}
    virtual ~DynamicModelLTV() = default;

    /** Discrete time dynamics
     * @param dt Time step
     * @param x Vec_x State
     * @param v Vec_v Process noise
     * @return Vec_x
     */
    virtual Vec_x f_d(double dt, const Vec_x& x, const Vec_u& u = Vec_x::Zero(), const Vec_x& v = Vec_x::Zero()) const override
    {
        Mat_xx A_d = this->A_d(dt, x);
        Mat_xu B_d = this->B_d(dt, x);
        Mat_xv G_d = this->G_d(dt, x);
        return A_d * x + B_d * u + G_d * v;
    }

    /** System matrix (Jacobian of discrete time dynamics with respect to state)
     * @param dt Time step
     * @param x Vec_x
     * @return State_jac
     */
    virtual Mat_xx A_d(double dt, const Vec_x& x) const = 0;

    /** Input matrix
     * @param dt Time step
     * @param x Vec_x
     * @return Input_jac
     */
    virtual Mat_xu B_d(double dt, const Vec_x& x) const 
    { 
        (void)dt; // unused
        (void)x; // unused
        return Mat_xu::Zero(); 
    }

    /** Process noise matrix (Jacobian of discrete time dynamics with respect to process noise)
     * @param dt Time step
     * @param x Vec_x
     * @return Process_noise_jac
     */
    virtual Mat_xv G_d(double dt, const Vec_x& x) const 
    { 
        (void)dt; // unused
        (void)x; // unused
        return Mat_xv::Identity(); 
    }

    /** Discrete time process noise covariance matrix
     * @param dt Time step
     * @param x Vec_x State
     */
    virtual Mat_vv Q_d(double dt, const Vec_x& x) const override = 0;

    /** Get the predicted state distribution given a state estimate
     * @param dt Time step
     * @param x_est Vec_x estimate
     * @return Vec_x
     */
    Gauss_x pred_from_est(double dt, const Gauss_x& x_est) const
    {
        Mat_xx P = x_est.cov();
        Mat_xx F_d = this->A_d(dt, x_est.mean());
        Mat_vv Q_d = this->Q_d(dt, x_est.mean());
        Mat_xv G_d = this->G_d(dt, x_est.mean());

        Gauss_x x_est_pred(f_d(dt, x_est.mean()), F_d * P * F_d.transpose() + G_d * Q_d * G_d.transpose());

        return x_est_pred;
    }

    /** Get the predicted state distribution given a state
     * @param dt Time step
     * @param x Vec_x
     * @return Vec_x
     */
    Gauss_x pred_from_state(double dt, const Vec_x& x) const
    {
        Mat_xx Q_d = this->Q_d(dt, x);
        Mat_xv G_d = this->G_d(dt, x);

        Gauss_x x_est_pred(this->f_d(dt, x), G_d * Q_d * G_d.transpose());

        return x_est_pred;
    }
};

/** Continuous Time Linear Time Varying Dynamic Model Interface. It uses excact discretization for everything. So it might be slow.
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension (Default: n_dim_x)
 * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
 * @note See https://en.wikipedia.org/wiki/Discretization#Discretization_of_process_noise for more info
 */
template <int n_dim_x, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x>
class DynamicModelCTLTV : public DynamicModelCT<n_dim_x, n_dim_u, n_dim_v>, public DynamicModelLTV<n_dim_x, n_dim_u, n_dim_v> {
public:
    using BaseI = DynamicModelI<n_dim_x, n_dim_u, n_dim_v>;
    static constexpr int N_DIM_x = n_dim_x;
    static constexpr int N_DIM_u = n_dim_u;
    static constexpr int N_DIM_v = n_dim_v;

    using Vec_x = typename BaseI::Vec_x;
    using Vec_u = typename BaseI::Vec_u;
    using Vec_v = typename BaseI::Vec_v;

    using Mat_xx = typename BaseI::Mat_xx;
    using Mat_xu = typename BaseI::Mat_xu;
    using Mat_xv = typename BaseI::Mat_xv;

    using Mat_uu = typename BaseI::Mat_uu;
    using Mat_vv = typename BaseI::Mat_vv;

    /** Continuous Time Linear Time Varying Dynamic Model Interface
     * @tparam n_dim_x  State dimension
     * @tparam n_dim_u  Input dimension (Default: n_dim_x)
     * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
     */
    DynamicModelCTLTV() : DynamicModelCT<n_dim_x, n_dim_u, n_dim_v>(), DynamicModelLTV<n_dim_x, n_dim_u, n_dim_v>() {}
    virtual ~DynamicModelCTLTV() = default;

    /** Continuous time dynamics
     * @param x Vec_x State
     * @param u Vec_u Input
     * @param v Vec_v Process noise
     * @return Vec_x State_dot
     */
    virtual Vec_x f_c(const Vec_x& x, const Vec_u& u = Vec_u::Zero(), const Vec_v& v = Vec_v::Zero()) const override = 0;


    /** System matrix (Jacobian of continuous time dynamics with respect to state)
     * @param x Vec_x State
     * @return State_jac
     */
    virtual Mat_xx A_c(const Vec_x& x) const = 0;

    /** Input matrix
     * @param x Vec_x State
     * @return Input_jac
     */
    virtual Mat_xu B_c(const Vec_x& x) const 
    { 
        (void)x; // unused
        return Mat_xu::Zero(); 
    }

    /** Process noise matrix
     * @return Process_noise_jac
     */
    virtual Mat_vv G_c() const { return Mat_vv::Identity(); }

    /** Process noise covariance matrix
     * @param x Vec_x State
     */
    virtual Mat_vv Q_c(const Vec_x& x) const = 0;



    /** Discrete time dynamics
     * @param dt Time step
     * @param x Vec_x State
     * @param v Vec_v Process noise
     * @return Vec_x
     */
    Vec_x f_d(double dt, const Vec_x& x, const Vec_u& u = Vec_x::Zero(), const Vec_x& v = Vec_x::Zero()) const override
    {
        Mat_xx A_d = this->A_d(dt, x);
        Mat_xu B_d = this->B_d(dt, x);
        Mat_xv G_d = this->G_d(dt, x);
        return A_d * x + B_d * u + G_d * v;
    }


    /** System dynamics (Jacobian of discrete time dynamics w.r.t. state). Using exact discretization.
     * @param dt Time step
     * @param x Vec_x
     * @return State_jac
     */
    Mat_xx A_d(double dt, const Vec_x& x) const override
    {
        return (A_c(x) * dt).exp();
    }

    /** Input matrix (Jacobian of discrete time dynamics w.r.t. input). Using exact discretization.
     * @param dt Time step
     * @param x Vec_x
     * @return Input_jac
     */
    Mat_xu B_d(double dt, const Vec_x& x) const override
    {
        return A_c(x).inverse() * (A_d(dt, x) - Mat_xx::Identity()) * B_c(x);
    }

    /** Process noise matrix (Jacobian of discrete time dynamics w.r.t. process noise). Using exact discretization.
     * @param dt Time step
     * @param x Vec_x
     * @return Process_noise_jac
     */
    Mat_xv G_d(double dt, const Vec_x& x) const override
    {
        return A_c(x).inverse() * (A_d(dt, x) - Mat_xx::Identity()) * G_c();
    }

    /** Discrete time process noise covariance matrix
     * Overriding DynamicModelI::Q_d
     * @param dt Time step
     * @param x Vec_x
     * @return Matrix Process noise covariance
     */
    Mat_xx Q_d(double dt, const Vec_x& x) const override
    {
        // See https://en.wikipedia.org/wiki/Discretization#Discretization_of_process_noise for more info

        Mat_xx A_c = this->A_c(x);
        Mat_xx Q_c = this->Q_c(x);

        Eigen::Matrix<double, 2 * N_DIM_x, 2 * N_DIM_x> van_loan_matrix;
        van_loan_matrix << -A_c, Q_c, Mat_xx::Zero(), A_c.transpose();
        van_loan_matrix *= dt;
        van_loan_matrix = van_loan_matrix.exp();
        Mat_xx F_d = van_loan_matrix.template block<N_DIM_x, N_DIM_x>(N_DIM_x, N_DIM_x).transpose();
        Mat_xx F_d_inv_Q_d = van_loan_matrix.template block<N_DIM_x, N_DIM_x>(0, N_DIM_x);
        Mat_xx Q_d = F_d * F_d_inv_Q_d;

        return Q_d;
    }
};

}  // namespace models
}  // namespace vortex
