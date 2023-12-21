/**
 * @file dynamic_model.hpp
 * @author Eirik Kolås
 * @brief Dynamic model interface. Based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-10-26
 */
#pragma once
#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions> // For exp
#include <functional>
#include <random>
#include <vortex_filtering/numerical_integration/erk_methods.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace models {
namespace interface {

/** Interface for dynamic models with dynamic size dimensions.
 * The purpose of this class is to provide a common interface for dynamic models of any dimension so that they can be used in the same way.
 * This class is not meant to be inherited from. Use DynamicModelI unless you need dynamic size dimensions.
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_dX
 * @note - Q_dX
 */
class DynamicModelX {
public:
  // Using dynamic Eigen types
  using VecX   = Eigen::VectorXd;
  using MatXX  = Eigen::MatrixXd;
  using GaussX = prob::MultiVarGauss<Eigen::Dynamic>;

  // Constructor to initialize the dimensions
  DynamicModelX(int dim_x, int dim_u, int dim_v) : dim_x_(dim_x), dim_u_(dim_u), dim_v_(dim_v) {}

  virtual ~DynamicModelX() = default;

  // Discrete-time dynamics
  virtual VecX f_dX(double dt, const VecX &x, const VecX &u, const VecX &v) const = 0;

  // Discrete-time process noise
  virtual MatXX Q_dX(double dt, const VecX &x) const = 0;

  // Sample from the discrete time dynamics
  virtual VecX sample_f_dX(double dt, const VecX &x, const VecX &u, std::mt19937 &gen) const = 0;

  // Sample from the discrete time dynamics
  virtual VecX sample_f_dX(double dt, const VecX &x) const = 0;

  int get_dim_x() const { return dim_x_; }
  int get_dim_u() const { return dim_u_; }
  int get_dim_v() const { return dim_v_; }

protected:
  const int dim_x_; // State dimension
  const int dim_u_; // Input dimension
  const int dim_v_; // Process noise dimension
};

/**
 * @brief Interface for dynamic models with static size dimensions
 *
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_d
 * @note - Q_d
 */
template <int n_dim_x, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x> class DynamicModelI : public DynamicModelX {
public:
  // Declare all sizes and types so that children of this class can reference them
  static constexpr int N_DIM_x = n_dim_x;
  static constexpr int N_DIM_u = n_dim_u;
  static constexpr int N_DIM_v = n_dim_v;

  using BaseX = DynamicModelX;
  using BaseI = DynamicModelI<N_DIM_x, N_DIM_u, N_DIM_v>;

  using Vec_x = Eigen::Vector<double, N_DIM_x>;
  using Vec_u = Eigen::Vector<double, N_DIM_u>;
  using Vec_v = Eigen::Vector<double, N_DIM_v>;

  using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
  using Mat_xu = Eigen::Matrix<double, N_DIM_x, N_DIM_u>;
  using Mat_xv = Eigen::Matrix<double, N_DIM_x, N_DIM_v>;

  using Mat_ux = Eigen::Matrix<double, N_DIM_u, N_DIM_x>;
  using Mat_uu = Eigen::Matrix<double, N_DIM_u, N_DIM_u>;
  using Mat_uv = Eigen::Matrix<double, N_DIM_u, N_DIM_v>;

  using Mat_vx = Eigen::Matrix<double, N_DIM_v, N_DIM_x>;
  using Mat_vv = Eigen::Matrix<double, N_DIM_v, N_DIM_v>;
  using Mat_vu = Eigen::Matrix<double, N_DIM_v, N_DIM_u>;

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
  virtual Vec_x f_d(double dt, const Vec_x &x, const Vec_u &u, const Vec_v &v) const = 0;

  /** Discrete time process noise covariance matrix
   * @param dt Time step
   * @param x Vec_x State
   */
  virtual Mat_vv Q_d(double dt, const Vec_x &x) const = 0;

  /** Sample from the discrete time dynamics
   * @param dt Time step
   * @param x Vec_x State
   * @param u Vec_u Input
   * @param gen Random number generator (For deterministic behaviour)
   * @return Vec_x Next state
   */
  Vec_x sample_f_d(double dt, const Vec_x &x, const Vec_u &u, std::mt19937 &gen) const
  {
    Gauss_v v = {Vec_v::Zero(), Q_d(dt, x)};
    return f_d(dt, x, u, v.sample(gen));
  }

  /** Sample from the discrete time dynamics
   * @param dt Time step
   * @param x Vec_x State
   * @return Vec_x Next state
   */
  Vec_x sample_f_d(double dt, const Vec_x &x) const
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    return sample_f_d(dt, x, Vec_u::Zero(), gen);
  }

private:
  // Override dynamic size functions to use static size functions
  BaseX::VecX f_dX(double dt, const BaseX::VecX &x, const BaseX::VecX &u, const BaseX::VecX &v) const override
  {
    Vec_x x_fixed = x;
    Vec_u u_fixed = u;
    Vec_v v_fixed = v;
    return f_d(dt, x_fixed, u_fixed, v_fixed);
  }

  BaseX::MatXX Q_dX(double dt, const BaseX::VecX &x) const override
  {
    Vec_x x_fixed = x;
    return Q_d(dt, x_fixed);
  }

  BaseX::VecX sample_f_dX(double dt, const BaseX::VecX &x, const BaseX::VecX &u, std::mt19937 &gen) const override
  {
    Vec_x x_fixed = x;
    Vec_u u_fixed = u;
    return sample_f_d(dt, x_fixed, u_fixed, gen);
  }

  BaseX::VecX sample_f_dX(double dt, const BaseX::VecX &x) const override
  {
    Vec_x x_fixed = x;
    return sample_f_d(dt, x_fixed);
  }
};

/**
 * @brief Continuous Time Dynamic Model Interface. x_dot = f_c(x, u, v)
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension (Default: n_dim_x)
 * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_c
 * @note - f_d (optional. Does a RK4 integration of f_c by default)
 * @note - Q_d
 */
template <int n_dim_x, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x> class DynamicModelCT : public DynamicModelI<n_dim_x, n_dim_u, n_dim_v> {
public:
  using BaseI                  = DynamicModelI<n_dim_x, n_dim_u, n_dim_v>;
  static constexpr int N_DIM_x = BaseI::N_DIM_x;
  static constexpr int N_DIM_u = BaseI::N_DIM_u;
  static constexpr int N_DIM_v = BaseI::N_DIM_v;

  using Vec_x = typename BaseI::Vec_x;
  using Vec_u = typename BaseI::Vec_u;
  using Vec_v = typename BaseI::Vec_v;

  using Mat_xx = typename BaseI::Mat_xx;
  using Mat_xu = typename BaseI::Mat_xu;
  using Mat_xv = typename BaseI::Mat_xv;

  using Mat_uu = typename BaseI::Mat_uu;
  using Mat_vv = typename BaseI::Mat_vv;

  using Dyn_mod_func = std::function<Vec_x(double t, const Vec_x &x)>;

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
  virtual Vec_x f_c(const Vec_x &x, const Vec_u &u, const Vec_v &v) const = 0;

  /** Discrete time process noise covariance matrix
   * @param dt Time step
   * @param x Vec_x State
   */
  virtual Mat_vv Q_d(double dt, const Vec_x &x) const override = 0;

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
  virtual Vec_x f_d(double dt, const Vec_x &x, const Vec_u &u, const Vec_v &v) const override
  {
    Dyn_mod_func f_c = [this, &u, &v](double, const Vec_x &x) { return this->f_c(x, u, v); };
    return vortex::integrator::RK4<N_DIM_x>::integrate(f_c, dt, x);
  }
};

/** Linear Time Variant Dynamic Model Interface. [x_k+1 = f_d = A_k*x_k + B_k*u_k + G_k*v_k]
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_d (optional)
 * @note - A_d
 * @note - B_d (optional)
 * @note - Q_d
 * @note - G_d (optional if n_dim_x == n_dim_v)
 */
template <int n_dim_x, int n_dim_u, int n_dim_v> class DynamicModelLTV : public DynamicModelI<n_dim_x, n_dim_u, n_dim_v> {
public:
  using BaseI                  = DynamicModelI<n_dim_x, n_dim_u, n_dim_v>;
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

  /** Linear Time Variant Dynamic Model Interface. [x_k+1 = f_d = A_k*x_k + B_k*u_k + G_k*v_k]
   * @tparam n_dim_x  State dimension
   * @tparam n_dim_u  Input dimension
   * @tparam n_dim_v  Process noise dimension
   * @note To derive from this class, you must override the following virtual functions:
   * @note - f_d (optional)
   * @note - A_d
   * @note - B_d (optional)
   * @note - Q_d
   * @note - G_d (optional if n_dim_x == n_dim_v)
   */
  DynamicModelLTV() : BaseI() {}
  virtual ~DynamicModelLTV() = default;

  /** Discrete time dynamics
   * @param dt Time step
   * @param x Vec_x State
   * @param u Vec_u Input
   * @param v Vec_v Process noise
   * @return Vec_x
   */
  virtual Vec_x f_d(double dt, const Vec_x &x, const Vec_u &u = Vec_u::Zero(), const Vec_v &v = Vec_v::Zero()) const override
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
  virtual Mat_xx A_d(double dt, const Vec_x &x) const = 0;

  /** Input matrix
   * @param dt Time step
   * @param x Vec_x
   * @return Input_jac
   */
  virtual Mat_xu B_d(double dt, const Vec_x &x) const
  {
    (void)dt; // unused
    (void)x;  // unused
    return Mat_xu::Zero();
  }

  /** Discrete time process noise covariance matrix
   * @param dt Time step
   * @param x Vec_x State
   */
  virtual Mat_vv Q_d(double dt, const Vec_x &x) const override = 0;

  /** Discrete time process noise transition matrix
   * @param dt Time step
   * @param x Vec_x State
   */
  virtual Mat_xv G_d(double dt, const Vec_x &x) const
  {
    if (N_DIM_x != N_DIM_v) {
      throw std::runtime_error("G_d not implemented");
    }
    (void)dt; // unused
    (void)x;  // unused
    return Mat_xv::Identity();
  }

  /** Get the predicted state distribution given a state estimate
   * @param dt Time step
   * @param x_est Vec_x estimate
   * @return Vec_x
   */
  Gauss_x pred_from_est(double dt, const Gauss_x &x_est, const Vec_u &u = Vec_u::Zero()) const
  {
    Mat_xx P      = x_est.cov();
    Mat_xx A_d    = this->A_d(dt, x_est.mean());
    Mat_xx GQGT_d = this->GQGT_d(dt, x_est.mean());

    Gauss_x x_est_pred(f_d(dt, x_est.mean(), u), A_d * P * A_d.transpose() + GQGT_d);
    return x_est_pred;
  }

  /** Get the predicted state distribution given a state
   * @param dt Time step
   * @param x Vec_x
   * @return Vec_x
   */
  Gauss_x pred_from_state(double dt, const Vec_x &x, const Vec_u &u = Vec_u::Zero()) const
  {
    Gauss_x x_est_pred(f_d(dt, x, u), GQGT_d(dt, x));
    return x_est_pred;
  }

protected:
  /** Process noise covariance matrix. For expressing the process noise in the state space.
   * @param dt Time step
   * @param x Vec_x State
   * @return Process_noise_jac
   */
  virtual Mat_xx GQGT_d(double dt, const Vec_x &x) const
  {
    Mat_vv Q_d = this->Q_d(dt, x);
    Mat_xv G_d = this->G_d(dt, x);

    return G_d * Q_d * G_d.transpose();
  }
};

/** Continuous Time Linear Time Varying Dynamic Model Interface. It uses exact discretization for everything. So it might be slow.
 * [x_dot = A_c*x + B_c*u + G_c*v]
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension (Default: n_dim_x)
 * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
 * @note See https://en.wikipedia.org/wiki/Discretization for more info on how the discretization is done.
 * @note To derive from this class, you must override the following functions:
 * @note - A_c
 * @note - B_c (optional)
 * @note - Q_c
 * @note - G_c (optional if n_dim_x == n_dim_v)
 */
template <int n_dim_x, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x> class DynamicModelCTLTV : public DynamicModelLTV<n_dim_x, n_dim_u, n_dim_v> {
public:
  using BaseI                  = DynamicModelI<n_dim_x, n_dim_u, n_dim_v>;
  static constexpr int N_DIM_x = n_dim_x;
  static constexpr int N_DIM_u = n_dim_u;
  static constexpr int N_DIM_v = n_dim_v;

  using Vec_x = typename BaseI::Vec_x;
  using Vec_u = typename BaseI::Vec_u;
  using Vec_v = typename BaseI::Vec_v;

  using Mat_xx = typename BaseI::Mat_xx;
  using Mat_xu = typename BaseI::Mat_xu;
  using Mat_xv = typename BaseI::Mat_xv;

  using Mat_ux = typename BaseI::Mat_ux;
  using Mat_uu = typename BaseI::Mat_uu;
  using Mat_vv = typename BaseI::Mat_vv;
  using Mat_vx = typename BaseI::Mat_vx;

  /** Continuous Time Linear Time Varying Dynamic Model Interface. [x_dot = A_c*x + B_c*u + G_c*v]
   * @tparam n_dim_x  State dimension
   * @tparam n_dim_u  Input dimension (Default: n_dim_x)
   * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
   */
  DynamicModelCTLTV() : DynamicModelLTV<n_dim_x, n_dim_u, n_dim_v>() {}
  virtual ~DynamicModelCTLTV() = default;

  /** Continuous time dynamics
   * @param x Vec_x State
   * @param u Vec_u Input
   * @param v Vec_v Process noise
   * @return Vec_x State_dot
   */
  Vec_x f_c(const Vec_x &x, const Vec_u &u = Vec_u::Zero(), const Vec_v &v = Vec_v::Zero()) const
  {
    Mat_xx A_c = this->A_c(x);
    Mat_xu B_c = this->B_c(x);
    Mat_xv G_c = this->G_c(x);
    return A_c * x + B_c * u + G_c * v;
  }

  /** System matrix (Jacobian of continuous time dynamics with respect to state)
   * @param x Vec_x State
   * @return State_jac
   */
  virtual Mat_xx A_c(const Vec_x &x) const = 0;

  /** Input matrix
   * @param x Vec_x State
   * @return Input_jac
   */
  virtual Mat_xu B_c(const Vec_x &x) const
  {
    (void)x; // unused
    return Mat_xu::Zero();
  }

  /** Process noise transition matrix. For expressing the process noise in the state space.
   * @param x Vec_x State
   * @return Process_noise_jac
   */
  virtual Mat_xv G_c(const Vec_x &x) const
  {
    if (N_DIM_x != N_DIM_v) {
      throw std::runtime_error("G_c not implemented");
    }
    (void)x; // unused
    return Mat_xv::Identity();
  }

  /** Process noise covariance matrix. This has the same dimension as the process noise.
   * @param x Vec_x State
   */
  virtual Mat_vv Q_c(const Vec_x &x) const = 0;

  /** System dynamics (Jacobian of discrete time dynamics w.r.t. state). Using exact discretization.
   * @param dt Time step
   * @param x Vec_x
   * @return State_jac
   */
  Mat_xx A_d(double dt, const Vec_x &x) const override { return (A_c(x) * dt).exp(); }

  /** Input matrix (Jacobian of discrete time dynamics w.r.t. input). Using exact discretization.
   * @param dt Time step
   * @param x Vec_x
   * @return Input_jac
   */
  Mat_xu B_d(double dt, const Vec_x &x) const override
  {
    Eigen::Matrix<double, N_DIM_x + N_DIM_u, N_DIM_x + N_DIM_u> van_loan;
    van_loan << A_c(x), B_c(x), Mat_ux::Zero(), Mat_uu::Zero();
    van_loan *= dt;
    van_loan = van_loan.exp();

    // Mat_xx A_d = van_loan.template block<N_DIM_x, N_DIM_x>(0, 0);
    Mat_xu B_d = van_loan.template block<N_DIM_x, N_DIM_u>(0, N_DIM_x);
    return B_d;
  }

  Mat_xv G_d(double dt, const Vec_x &x) const override
  {
    Eigen::Matrix<double, N_DIM_x + N_DIM_v, N_DIM_x + N_DIM_v> van_loan;
    van_loan << A_c(x), G_c(x), Mat_vx::Zero(), Mat_vv::Zero();
    van_loan *= dt;
    van_loan = van_loan.exp();

    // Mat_xx A_d = van_loan.template block<N_DIM_x, N_DIM_x>(0, 0);
    Mat_xv G_d = van_loan.template block<N_DIM_x, N_DIM_v>(0, N_DIM_x);
    return G_d;
  }

  /** Discrete time process noise covariance matrix. This is super scuffed, but it works... (As long as G_d^T*G_d is invertible)
   * Overriding DynamicModelI::Q_d
   * @param dt Time step
   * @param x Vec_x
   * @return Matrix Process noise covariance
   */
  Mat_vv Q_d(double dt, const Vec_x &x) const override
  {
    Mat_xx GQGT_d = this->GQGT_d(dt, x);
    Mat_xv G_d    = this->G_d(dt, x);
    // psuedo inverse of G_d
    Mat_vx G_d_pinv = G_d.completeOrthogonalDecomposition().pseudoInverse();

    return G_d_pinv * GQGT_d * G_d_pinv.transpose();
    return Mat_vv::Identity();
  }

  Mat_xx GQGT_d(double dt, const Vec_x &x) const override
  {
    Mat_xx A_c = this->A_c(x);
    Mat_vv Q_c = this->Q_c(x);
    Mat_xv G_c = this->G_c(x);

    Eigen::Matrix<double, 2 * N_DIM_x, 2 * N_DIM_x> van_loan;
    van_loan << -A_c, G_c * Q_c * G_c.transpose(), Mat_xx::Zero(), A_c.transpose();
    van_loan *= dt;
    van_loan              = van_loan.exp();
    Mat_xx A_d            = van_loan.template block<N_DIM_x, N_DIM_x>(N_DIM_x, N_DIM_x).transpose();
    Mat_xx A_d_inv_GQGT_d = van_loan.template block<N_DIM_x, N_DIM_x>(0, N_DIM_x); // A_d^(-1) * G * Q * G^T
    Mat_xx GQGT_d         = A_d * A_d_inv_GQGT_d;                                  // G * Q * G^T
    return GQGT_d;
  }
};

} // namespace interface
} // namespace models
} // namespace vortex
