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
#include <manif/manif.h>
#include <memory>
#include <random>
#include <tl/optional.hpp>
#include <vortex_filtering/numerical_integration/erk_methods.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/lie_group_gauss.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::models {
namespace interface {

using std::size_t;
/**
 * @brief Interface for dynamic models
 *
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_d
 * @note - Q_d
 */
template <size_t n_dim_x, size_t n_dim_u = n_dim_x, size_t n_dim_v = n_dim_x> class DynamicModel {
public:
  static constexpr int N_DIM_x = (int)n_dim_x;
  static constexpr int N_DIM_u = (int)n_dim_u;
  static constexpr int N_DIM_v = (int)n_dim_v;

  using T = Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  DynamicModel()          = default;
  virtual ~DynamicModel() = default;

  /** Discrete time dynamics
   * @param dt Time step
   * @param x T::Vec_x State
   * @param u T::Vec_u Input
   * @param v T::Vec_v Process noise
   * @return T::Vec_x Next state
   */
  virtual T::Vec_x f_d(double dt, const T::Vec_x &x, const T::Vec_u &u, const T::Vec_v &v) const = 0;

  /** Discrete time process noise covariance matrix
   * @param dt Time step
   * @param x T::Vec_x State
   */
  virtual T::Mat_vv Q_d(double dt, const T::Vec_x &x) const = 0;

  /** Sample from the discrete time dynamics
   * @param dt Time step
   * @param x T::Vec_x State
   * @param u T::Vec_u Input
   * @param gen Random number generator (For deterministic behaviour)
   * @return T::Vec_x Next state
   */
  T::Vec_x sample_f_d(double dt, const T::Vec_x &x, const T::Vec_u &u, std::mt19937 &gen) const
  {
    typename T::Gauss_v v = {T::Vec_v::Zero(), Q_d(dt, x)};
    return f_d(dt, x, u, v.sample(gen));
  }
};

/** Linear Time Variant Dynamic Model Interface. [x_k+1 = f_d = A_k*x_k + B_k*u_k + G_k*v_k]
 * @tparam n_dim_x  State dimension
 * @tparam n_dim_u  Input dimension (Default: n_dim_x)
 * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_d (optional)
 * @note - A_d
 * @note - B_d (optional)
 * @note - Q_d
 * @note - G_d (optional if n_dim_x == n_dim_v)
 */
template <size_t n_dim_x, size_t n_dim_u = n_dim_x, size_t n_dim_v = n_dim_x> class DynamicModelLTV : public DynamicModel<n_dim_x, n_dim_u, n_dim_v> {
public:
  static constexpr int N_DIM_x = n_dim_x;
  static constexpr int N_DIM_u = n_dim_u;
  static constexpr int N_DIM_v = n_dim_v;

  using T = Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  /** Linear Time Variant Dynamic Model Interface. [x_k+1 = f_d = A_k*x_k + B_k*u_k + G_k*v_k]
   * @tparam n_dim_x  State dimension
   * @tparam n_dim_u  Input dimension (Default: n_dim_x)
   * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
   * @note To derive from this class, you must override the following virtual functions:
   * @note - f_d (optional)
   * @note - A_d
   * @note - B_d (optional)
   * @note - Q_d
   * @note - G_d (optional if n_dim_x == n_dim_v)
   */
  DynamicModelLTV()
      : DynamicModel<N_DIM_x, N_DIM_u, N_DIM_v>()
  {
  }
  virtual ~DynamicModelLTV() = default;

  /** Discrete time dynamics
   * @param dt Time step
   * @param x Vec_x State
   * @param u Vec_u Input
   * @param v Vec_v Process noise
   * @return Vec_x
   */
  virtual T::Vec_x f_d(double dt, const T::Vec_x &x, const T::Vec_u &u = T::Vec_u::Zero(), const T::Vec_v &v = T::Vec_v::Zero()) const override
  {
    typename T::Mat_xx A_d = this->A_d(dt, x);
    typename T::Mat_xu B_d = this->B_d(dt, x);
    typename T::Mat_xv G_d = this->G_d(dt, x);
    return A_d * x + B_d * u + G_d * v;
  }

  /** System matrix (Jacobian of discrete time dynamics with respect to state)
   * @param dt Time step
   * @param x T::Vec_x
   * @return State_jac
   */
  virtual T::Mat_xx A_d(double dt, const T::Vec_x &x) const = 0;

  /** Input matrix
   * @param dt Time step
   * @param x T::Vec_x
   * @return Input_jac
   */
  virtual T::Mat_xu B_d(double dt, const T::Vec_x &x) const
  {
    (void)dt; // unused
    (void)x;  // unused
    return T::Mat_xu::Zero();
  }

  /** Discrete time process noise covariance matrix
   * @param dt Time step
   * @param x T::Vec_x State
   */
  virtual T::Mat_vv Q_d(double dt, const T::Vec_x &x) const override = 0;

  /** Discrete time process noise transition matrix
   * @param dt Time step
   * @param x T::Vec_x State
   */
  virtual T::Mat_xv G_d(double, const T::Vec_x &) const
  {
    if (this->N_DIM_x != this->N_DIM_v) {
      throw std::runtime_error("G_d not implemented");
    }
    return T::Mat_xv::Identity();
  }

  /** Get the predicted state distribution given a state estimate
   * @param dt Time step
   * @param x_est T::Vec_x estimate
   * @return T::Vec_x
   */
  auto pred_from_est(double dt, const auto &x_est, const T::Vec_u &u = T::Vec_u::Zero()) const -> std::remove_reference_t<decltype(x_est)>
    requires(concepts::MultiVarGaussLike<decltype(x_est), N_DIM_x>)
  {
    typename T::Mat_xx P      = x_est.cov();
    typename T::Mat_xx A_d    = this->A_d(dt, x_est.mean());
    typename T::Mat_xx GQGT_d = this->GQGT_d(dt, x_est.mean());

    return {f_d(dt, x_est.mean(), u), A_d * P * A_d.transpose() + GQGT_d};
  }

  /** Get the predicted state distribution given a state
   * @param dt Time step
   * @param x T::Vec_x
   * @return T::Vec_x
   */
  T::Gauss_x pred_from_state(double dt, const T::Vec_x &x, const T::Vec_u &u = T::Vec_u::Zero()) const
  {
    typename T::Gauss_x x_est_pred = {f_d(dt, x, u), GQGT_d(dt, x)};
    return x_est_pred;
  }

protected:
  /** Process noise covariance matrix. For expressing the process noise in the state space.
   * @param dt Time step
   * @param x T::Vec_x State
   * @return Process_noise_jac
   */
  virtual T::Mat_xx GQGT_d(double dt, const T::Vec_x &x) const
  {
    typename T::Mat_vv Q_d = this->Q_d(dt, x);
    typename T::Mat_xv G_d = this->G_d(dt, x);

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
template <size_t n_dim_x, size_t n_dim_u = n_dim_x, size_t n_dim_v = n_dim_x> class DynamicModelCTLTV : public DynamicModelLTV<n_dim_x, n_dim_u, n_dim_v> {
public:
  static constexpr int N_DIM_x = n_dim_x;
  static constexpr int N_DIM_u = n_dim_u;
  static constexpr int N_DIM_v = n_dim_v;

  using T = Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  /** Continuous Time Linear Time Varying Dynamic Model Interface. [x_dot = A_c*x + B_c*u + G_c*v]
   * @tparam n_dim_x  State dimension
   * @tparam n_dim_u  Input dimension (Default: n_dim_x)
   * @tparam n_dim_v  Process noise dimension (Default: n_dim_x)
   */
  DynamicModelCTLTV()
      : DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v>()
  {
  }
  virtual ~DynamicModelCTLTV() = default;

  /** Continuous time dynamics
   * @param x T::Vec_x State
   * @param u T::Vec_u Input
   * @param v T::Vec_v Process noise
   * @return T::Vec_x State_dot
   */
  T::Vec_x f_c(const T::Vec_x &x, const T::Vec_u &u = T::Vec_u::Zero(), const T::Vec_v &v = T::Vec_v::Zero()) const
  {
    typename T::Mat_xx A_c = this->A_c(x);
    typename T::Mat_xu B_c = this->B_c(x);
    typename T::Mat_xv G_c = this->G_c(x);
    return A_c * x + B_c * u + G_c * v;
  }

  /** System matrix (Jacobian of continuous time dynamics with respect to state)
   * @param x T::Vec_x State
   * @return State_jac
   */
  virtual T::Mat_xx A_c(const T::Vec_x &x) const = 0;

  /** Input matrix
   * @param x T::Vec_x State
   * @return Input_jac
   */
  virtual T::Mat_xu B_c(const T::Vec_x &x) const
  {
    (void)x; // unused
    return T::Mat_xu::Zero();
  }

  /** Process noise transition matrix. For expressing the process noise in the state space.
   * @param x T::Vec_x State
   * @return Process_noise_jac
   */
  virtual T::Mat_xv G_c(const T::Vec_x &x) const
  {
    if (N_DIM_x != N_DIM_v) {
      throw std::runtime_error("G_c not implemented");
    }
    (void)x; // unused
    return T::Mat_xv::Identity();
  }

  /** Process noise covariance matrix. This has the same dimension as the process noise.
   * @param x T::Vec_x State
   */
  virtual T::Mat_vv Q_c(const T::Vec_x &x) const = 0;

  /** System dynamics (Jacobian of discrete time dynamics w.r.t. state). Using exact discretization.
   * @param dt Time step
   * @param x T::Vec_x
   * @return State_jac
   */
  T::Mat_xx A_d(double dt, const T::Vec_x &x) const override { return (A_c(x) * dt).exp(); }

  /** Input matrix (Jacobian of discrete time dynamics w.r.t. input). Using exact discretization.
   * @param dt Time step
   * @param x T::Vec_x
   * @return Input_jac
   */
  T::Mat_xu B_d(double dt, const T::Vec_x &x) const override
  {
    Eigen::Matrix<double, N_DIM_x + N_DIM_u, N_DIM_x + N_DIM_u> van_loan;
    van_loan << A_c(x), B_c(x), T::Mat_ux::Zero(), T::Mat_uu::Zero();
    van_loan *= dt;
    van_loan = van_loan.exp();

    // T::Mat_xx A_d = van_loan.template block<N_DIM_x, N_DIM_x>(0, 0);
    typename T::Mat_xu B_d = van_loan.template block<N_DIM_x, N_DIM_u>(0, N_DIM_x);
    return B_d;
  }

  T::Mat_xv G_d(double dt, const T::Vec_x &x) const override
  {
    Eigen::Matrix<double, N_DIM_x + N_DIM_v, N_DIM_x + N_DIM_v> van_loan;
    van_loan.template topLeftCorner<N_DIM_x, N_DIM_x>()     = A_c(x);
    van_loan.template topRightCorner<N_DIM_x, N_DIM_v>()    = G_c(x);
    van_loan.template bottomLeftCorner<N_DIM_v, N_DIM_x>()  = T::Mat_vx::Zero();
    van_loan.template bottomRightCorner<N_DIM_v, N_DIM_v>() = T::Mat_vv::Zero();

    van_loan *= dt;
    van_loan = van_loan.exp();

    // T::Mat_xx A_d = van_loan.template block<N_DIM_x, N_DIM_x>(0, 0);
    typename T::Mat_xv G_d = van_loan.template block<N_DIM_x, N_DIM_v>(0, N_DIM_x);
    return G_d;
  }

  /** Discrete time process noise covariance matrix. This is super scuffed, but it works... (As long as G_d^T*G_d is invertible)
   * Overriding DynamicModel::Q_d
   * @param dt Time step
   * @param x T::Vec_x
   * @return Matrix Process noise covariance
   */
  T::Mat_vv Q_d(double dt, const T::Vec_x &x) const override
  {
    typename T::Mat_xx GQGT_d = this->GQGT_d(dt, x);
    typename T::Mat_xv G_d    = this->G_d(dt, x);
    // psuedo inverse of G_d
    typename T::Mat_vx G_d_pinv = G_d.completeOrthogonalDecomposition().pseudoInverse();

    return G_d_pinv * GQGT_d * G_d_pinv.transpose();
  }

  T::Mat_xx GQGT_d(double dt, const T::Vec_x &x) const override
  {
    typename T::Mat_xx A_c = this->A_c(x);
    typename T::Mat_vv Q_c = this->Q_c(x);
    typename T::Mat_xv G_c = this->G_c(x);

    Eigen::Matrix<double, 2 * N_DIM_x, 2 * N_DIM_x> van_loan;
    van_loan.template topLeftCorner<N_DIM_x, N_DIM_x>()     = -A_c;
    van_loan.template topRightCorner<N_DIM_x, N_DIM_x>()    = G_c * Q_c * G_c.transpose();
    van_loan.template bottomLeftCorner<N_DIM_x, N_DIM_x>()  = T::Mat_xx::Zero();
    van_loan.template bottomRightCorner<N_DIM_x, N_DIM_x>() = A_c.transpose();

    van_loan *= dt;
    van_loan = van_loan.exp();

    typename T::Mat_xx A_d            = van_loan.template block<N_DIM_x, N_DIM_x>(N_DIM_x, N_DIM_x).transpose();
    typename T::Mat_xx A_d_inv_GQGT_d = van_loan.template block<N_DIM_x, N_DIM_x>(0, N_DIM_x); // A_d^(-1) * G * Q * G^T
    typename T::Mat_xx GQGT_d         = A_d * A_d_inv_GQGT_d;                                  // G * Q * G^T
    return GQGT_d;
  }
};

/**
 * @brief Interface for dynamic models on Lie groups
 *
 * @tparam Derived The Lie group type from manif (e.g., SE3d, SO3d)
 * @tparam n_dim_u  Input dimension
 * @tparam n_dim_v  Process noise dimension
 * @note To derive from this class, you must override the following virtual functions:
 * @note - f_d
 * @note - Q_d
 */
template <typename Type_x, typename Type_u, typename Type_v> class LieGroupDynamicModel {
public:
  using Mx = Type_x; // Lie group of the state
  using Mu = Type_u; // Lie group of the input
  using Mv = Type_v; // Lie group of the process noise

  static constexpr int N_DIM_x = Mx::DoF;
  static constexpr int N_DIM_u = Mu::DoF;
  static constexpr int N_DIM_v = Mv::DoF;

  using T = Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  using Tx = typename Mx::Tangent; // Tangent space of the Lie group of the state
  using Tu = typename Mu::Tangent; // Tangent space of the Lie group of the input
  using Tv = typename Mv::Tangent; // Tangent space of the Lie group of the process noise

  LieGroupDynamicModel()          = default;
  virtual ~LieGroupDynamicModel() = default;

  /** Discrete time dynamics on the Lie group
   * @param dt Time step
   * @param x State (Lie group element)
   * @param u Input vector
   * @param v Process noise in the tangent space
   * @return Next state on the Lie group
   */
  virtual Mx f_d(double dt, const Mx &x, const Mu &u, const Mv &v) const = 0;

  /** Discrete time process noise covariance matrix in the tangent space
   * @param dt Time step
   * @param x Mx (Lie group element)
   * @return Covariance matrix in tangent space of the process noise
   */
  virtual T::Mat_vv Q_d(double dt, const Mx &x) const = 0;

  /** Sample from the discrete time dynamics on the Lie group
   * @param dt Time step
   * @param x State (Lie group element)
   * @param u Input vector
   * @param gen Random number generator
   * @return Sampled next state on the Lie group
   */
  Mx sample_f_d(double dt, const Mx &x, const Mu &u, std::mt19937 &gen) const
  {
    prob::LieGroupGauss<Mv> noise_model(Mv::Identity(), Q_d(dt, x));
    Mv sampled_noise = noise_model.sample(gen);
    return f_d(dt, x, u, sampled_noise);
  }
};

template <typename Type_x, typename Type_u, typename Type_v>
class LieGroupDynamicModelLTV : public LieGroupDynamicModel<Type_x, Type_u, Type_v> {
public:
  using Mx = Type_x;
  using Mu = Type_u;
  using Mv = Type_v;

  using Tx = typename Mx::Tangent;

  using T = Types_xuv<Mx::DoF, Mu::DoF, Mv::DoF>;

  using Gauss_x = vortex::prob::LieGroupGauss<Mx>;

  /**
   * @brief
   *
   * @param dt Time step
   * @param x State (Lie group element)
   * @param tau_x Perturbation in the tangent space of the state
   * @param u Input vector
   * @param v Process noise in the tangent space
   * @return Mx Next state on the Lie group
   */
  Mx f_d(double dt, const Mx &x, const Mu &u, const Mv &v) const = 0;

  virtual T::Mat_xx J_f_x(double dt, const Mx &x) const = 0;

  virtual T::Mat_xu J_f_u(double /*dt*/, const Mx & /*x*/) const { return T::Mat_xu::Zero(); }

  virtual T::Mat_xv J_f_v(double /*dt*/, const Mx & /*x*/) const { return T::Mat_xv::Identity(); }

  Gauss_x pred_from_est(double dt, const Gauss_x &x_est, const Mu &u) const
  {
    auto mean            = x_est.mean();
    typename T::Mat_xx P = x_est.cov();
    typename T::Mat_xx A = J_f_x(dt, mean);
    typename T::Mat_xv G = J_f_v(dt, mean);
    typename T::Mat_xx Q = G * this->Q_d(dt, mean) * G.transpose();
    return {f_d(dt, mean, u, Mv::Identity()), A * P * A.transpose() + Q};
  }

  Gauss_x pred_from_state(double dt, const Mx &x, const Mu &u) const
  {
    typename T::Mat_xx Q_v = this->Q_d(dt, x);
    typename T::Mat_xv G   = J_f_v(dt, x);
    typename T::Mat_xx Q_x = G * Q_v * G.transpose();
    return {f_d(dt, x, u, Mv::Identity()), Q_x};
  }
};

} // namespace interface

} // namespace vortex::models