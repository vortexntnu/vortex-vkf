#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/imm_model.hpp>

namespace vortex {
namespace models {

constexpr int UNUSED = 1; // For when a template parameter is required but not used.

/** Identity Dynamic Model
 * @tparam n_dim_x Number of dimensions in state vector
 */
template <size_t n_dim_x> class IdentityDynamicModel : public interface::DynamicModelLTV<n_dim_x> {
public:
  using DynModI = interface::DynamicModelLTV<n_dim_x>;
  using Vec_x   = typename DynModI::Vec_x;
  using Mat_xx  = typename DynModI::Mat_xx;
  using Mat_xv  = typename DynModI::Mat_xv;
  using Mat_vv  = typename DynModI::Mat_vv;

  /** Identity Dynamic Model
   * @param std Standard deviation of process noise
   */
  IdentityDynamicModel(double std)
      : Q_(Mat_xx::Identity() * std * std)
  {
  }

  /** Identity Dynamic Model
   * @param Q Process noise covariance
   */
  IdentityDynamicModel(Mat_vv Q)
      : Q_(Q)
  {
  }

  Mat_xx A_d(double dt, const Vec_x /*x*/ &) const override { return Mat_xx::Identity() * dt; }
  Mat_vv Q_d(double /*dt*/, const Vec_x /*x*/ &) const override { return Q_; }

private:
  Mat_vv Q_;
};

/** (Nearly) Constant Position Model
 * State x = [pos], where pos is a `n_spatial_dim`-dimensional vector
 * @tparam n_spatial_dim Number of spatial dimensions
 */
class ConstantPosition : public interface::DynamicModelLTV<2, UNUSED, 2> {
public:
  static constexpr int N_STATES = 2;
  using DynModI                 = interface::DynamicModelLTV<N_STATES, UNUSED, N_STATES>;
  using typename DynModI::Vec_v;
  using typename DynModI::Vec_x;

  using typename DynModI::Mat_vv;
  using typename DynModI::Mat_xv;
  using typename DynModI::Mat_xx;

  using Vec_s  = Eigen::Vector<double, N_STATES>;
  using Mat_ss = Eigen::Matrix<double, N_STATES, N_STATES>;

  using ST = StateType;
  static constexpr std::array<ST, N_STATES> StateNames{ST::position, ST::position};

  /** Constant Position Model in 2D
   * x = [x, y]
   * @param std_pos Standard deviation of position
   */
  ConstantPosition(double std_pos)
      : std_pos_(std_pos)
  {
  }

  /** Get the Jacobian of the continuous state transition model with respect to the state.
   * @param dt Time step
   * @param x State (unused)
   * @return Mat_xx
   * @note Overriding DynamicModelLTV::A_d
   */
  Mat_xx A_d(double /*dt*/, const Vec_x & = Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    return I;
  }

  /** Get the Jacobian of the continuous state transition model with respect to the process noise.
   * @param dt Time step
   * @param x State (unused)
   * @return Mat_xv
   * @note Overriding DynamicModelLTV::G_d
   */
  Mat_xv G_d(double dt, const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    return 0.5 * dt * I;
  }

  /** Get the continuous time process noise covariance matrix.
   * @param dt Time step (unused)
   * @param x State (unused)
   * @return Mat_xx Process noise covariance
   * @note Overriding DynamicModelLTV::Q_d
   */
  Mat_vv Q_d(double /*dt*/ = 0.0, const Vec_x /*x*/ & = Vec_x::Zero()) const override { return Mat_vv::Identity() * std_pos_ * std_pos_; }

private:
  double std_pos_;
};

/** (Nearly) Constant Velocity Model.
 * State x = [pos, vel], where pos and vel are `n_spatial_dim`-dimensional vectors
 * @tparam n_spatial_dim Number of spatial dimensions
 */
class ConstantVelocity : public interface::DynamicModelLTV<4, UNUSED, 2> {
public:
  static constexpr int N_SPATIAL_DIM = 2;
  static constexpr int N_STATES      = 2 * N_SPATIAL_DIM;

  using DynModI = interface::DynamicModelLTV<N_STATES, UNUSED, N_SPATIAL_DIM>;
  using typename DynModI::Mat_vv;
  using typename DynModI::Mat_xv;
  using typename DynModI::Mat_xx;
  using typename DynModI::Vec_x;

  using Vec_s  = Eigen::Matrix<double, N_SPATIAL_DIM, 1>;
  using Mat_ss = Eigen::Matrix<double, N_SPATIAL_DIM, N_SPATIAL_DIM>;

  using ST = StateType;
  static constexpr std::array<ST, N_STATES> StateNames{ST::position, ST::position, ST::velocity, ST::velocity};

  /**
   * @brief Constant Velocity Model in 2D
   * x = [x, y, x_dot, y_dot]
   * @param std_vel Standard deviation of velocity
   */
  ConstantVelocity(double std_vel)
      : std_vel_(std_vel)
  {
  }

  /** Get the Jacobian of the continuous state transition model with respect to the state.
   * @param dt Time step
   * @param x State (unused)
   * @return Mat_xx
   * @note Overriding DynamicModelLTV::A_d
   */
  Mat_xx A_d(double dt, const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_ss O = Mat_ss::Zero();
    Mat_xx A;
    // clang-format off
    A << I, I*dt,
         O, I;
    // clang-format on
    return A;
  }

  /** Get the Jacobian of the continuous state transition model with respect to the process noise.
   * @param dt Time step
   * @param x State (unused)
   * @return Mat_xv
   * @note Overriding DynamicModelLTV::G_d
   */
  Mat_xv G_d(double dt, const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_xv G;
    // clang-format off
    G << 0.5*dt*dt*I,
                dt*I;
    // clang-format on
    return G;
  }

  /** Get the continuous time process noise covariance matrix.
   * @param dt Time step (unused)
   * @param x State (unused)
   * @return Mat_xx Process noise covariance
   * @note Overriding DynamicModelLTV::Q_d
   */
  Mat_vv Q_d(double /*dt*/ = 0.0, const Vec_x /*x*/ & = Vec_x::Zero()) const override { return Mat_vv::Identity() * std_vel_ * std_vel_; }

private:
  double std_vel_;
};

/** (Nearly) Constant Acceleration Model.
 * State vector x = [pos, vel, acc], where pos, vel and acc are `n_spatial_dim`-dimensional vectors
 * @tparam n_spatial_dim Number of spatial dimensions
 */
class ConstantAcceleration : public interface::DynamicModelLTV<3 * 2, UNUSED, 2 * 2> {
public:
  static constexpr int N_SPATIAL_DIM = 2;
  static constexpr int N_STATES      = 3 * N_SPATIAL_DIM;

  using DynModI = interface::DynamicModelLTV<N_STATES, UNUSED, 2 * N_SPATIAL_DIM>;
  using typename DynModI::Vec_v;
  using typename DynModI::Vec_x;

  using typename DynModI::Mat_vv;
  using typename DynModI::Mat_xv;
  using typename DynModI::Mat_xx;

  using Vec_s  = Eigen::Matrix<double, N_SPATIAL_DIM, 1>;
  using Mat_ss = Eigen::Matrix<double, N_SPATIAL_DIM, N_SPATIAL_DIM>;

  using ST = StateType;
  static constexpr std::array<ST, N_STATES> StateNames{ST::position, ST::position, ST::velocity, ST::velocity, ST::acceleration, ST::acceleration};
  /** Constant Acceleration Model
   * @param std_vel Standard deviation of velocity
   * @param std_acc Standard deviation of acceleration
   */
  ConstantAcceleration(double std_vel, double std_acc)
      : std_vel_(std_vel)
      , std_acc_(std_acc)
  {
  }

  /** Get the Jacobian of the continuous state transition model with respect to the state.
   * @param x State
   * @return Mat_xx
   * @note Overriding DynamicModelLTV::A_d
   */
  Mat_xx A_d(double dt, const Vec_x /*x*/ &) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_ss O = Mat_ss::Zero();
    Mat_xx A;
    // clang-format off
    A << I, I*dt, I*0.5*dt*dt,
         O, I   , I*dt       ,
         O, O   , I          ;
    // clang-format on
    return A;
  }

  Mat_xv G_d(double dt, const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_ss O = Mat_ss::Zero();
    Mat_xv G;
    // clang-format off
    G << I*dt, I*0.5*dt*dt,
         I   , I*dt       ,
         O   , I          ;
    // clang-format on
    return G;
  }

  /** Get the continuous time process noise covariance matrix.
   * @param dt Time step (unused)
   * @param x State
   * @return Mat_xx Process noise covariance
   * @note Overriding DynamicModelLTV::Q_d
   */
  Mat_vv Q_d(double /*dt*/ = 0.0, const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    Vec_v D;
    double var_vel = std_vel_ * std_vel_;
    double var_acc = std_acc_ * std_acc_;
    D << var_vel, var_vel, var_acc, var_acc;
    return D.asDiagonal();
  }

private:
  double std_vel_;
  double std_acc_;
};

/** Coordinated Turn Model in 2D.
 * x = [x_pos, y_pos, x_vel, y_vel, turn_rate]
 */
class CoordinatedTurn : public interface::DynamicModelCTLTV<5, UNUSED, 3> {
public:
  static constexpr int N_STATES = 5;
  using DynModI                 = interface::DynamicModelCTLTV<N_STATES, UNUSED, 3>;
  using typename DynModI::Vec_v;
  using typename DynModI::Vec_x;

  using typename DynModI::Mat_vv;
  using typename DynModI::Mat_xv;
  using typename DynModI::Mat_xx;

  using ST = StateType;
  static constexpr std::array<ST, N_STATES> StateNames{ST::position, ST::position, ST::velocity, ST::velocity, ST::turn_rate};
  /** (Nearly) Coordinated Turn Model in 2D. (Nearly constant speed, nearly constant turn rate)
   * State = [x, y, x_dot, y_dot, omega]
   * @param std_vel Standard deviation of velocity
   * @param std_turn Standard deviation of turn rate
   */
  CoordinatedTurn(double std_vel, double std_turn)
      : std_vel_(std_vel)
      , std_turn_(std_turn)
  {
  }

  /** Get the Jacobian of the continuous state transition model with respect to the state.
   * @param x State
   * @return Mat_xx
   * @note Overriding DynamicModelCTLTV::A_c
   */
  Mat_xx A_c(const Vec_x /*x*/ &x) const override
  {
    // clang-format off
    return Mat_xx{
      {0, 0, 1   , 0   , 0},
      {0, 0, 0   , 1   , 0},
      {0, 0, 0   ,-x(4), 0},
      {0, 0, x(4), 0   , 0},
      {0, 0, 0   , 0   , 0}
    };
    // clang-format on
  }

  /** Get the continuous time process noise matrix
   * @param x State (unused)
   * return Mat_xv Process noise matrix
   * @note Overriding DynamicModelCTLTV::G_c
   */
  Mat_xv G_c(const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    // clang-format off
    return Mat_xv {
     {0, 0, 0},
     {0, 0, 0},
     {1, 0, 0},
     {0, 1, 0},
     {0, 0, 1}
    };
    // clang-format on
  }

  /** Get the continuous time process noise covariance matrix.
   * @param x State
   * @return Mat_xx Process noise covariance
   * @note Overriding DynamicModelCTLTV::Q_c
   */
  Mat_vv Q_c(const Vec_x /*x*/ & = Vec_x::Zero()) const override
  {
    double var_vel  = std_vel_ * std_vel_;
    double var_turn = std_turn_ * std_turn_;

    return Vec_v{var_vel, var_vel, var_turn}.asDiagonal();
  }

private:
  double std_vel_;
  double std_turn_;
};

} // namespace models
} // namespace vortex