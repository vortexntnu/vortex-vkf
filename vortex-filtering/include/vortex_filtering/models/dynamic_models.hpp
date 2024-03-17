#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex {
namespace models {

constexpr int UNUSED = 1; // For when a template parameter is required but not used.

/** Identity Dynamic Model
 * @tparam n_dim_x Number of dimensions in state vector
 */
template <size_t n_dim_x> class IdentityDynamicModel : public interface::DynamicModelLTV<n_dim_x> {
  using Parent = interface::DynamicModelLTV<n_dim_x>;

public:
  static constexpr int N_DIM_x = Parent::N_DIM_x;
  static constexpr int N_DIM_u = Parent::N_DIM_u;
  static constexpr int N_DIM_v = Parent::N_DIM_v;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  /** Identity Dynamic Model
   * @param std Standard deviation of process noise
   */
  IdentityDynamicModel(double std)
      : Q_(T::Mat_xx::Identity() * std * std)
  {
  }

  /** Identity Dynamic Model
   * @param Q Process noise covariance
   */
  IdentityDynamicModel(T::Mat_vv Q)
      : Q_(Q)
  {
  }

  T::Mat_xx A_d(double dt, const T::Vec_x /*x*/ &) const override { return T::Mat_xx::Identity() * dt; }
  T::Mat_vv Q_d(double /*dt*/, const T::Vec_x /*x*/ &) const override { return Q_; }

private:
  T::Mat_vv Q_;
};

/** (Nearly) Constant Position Model
 * State x = [pos], where pos is a `n_spatial_dim`-dimensional vector
 * @tparam n_spatial_dim Number of spatial dimensions
 */
class ConstantPosition : public interface::DynamicModelLTV<2, UNUSED, 2> {
  using Parent = interface::DynamicModelLTV<2, UNUSED, 2>;

public:
  static constexpr int N_DIM_x = Parent::N_DIM_x;
  static constexpr int N_DIM_u = Parent::N_DIM_u;
  static constexpr int N_DIM_v = Parent::N_DIM_v;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;

  using ST = StateType;
  static constexpr std::array StateNames{ST::position, ST::position};

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
   * @return T::Mat_xx
   * @note Overriding DynamicModelLTV::A_d
   */
  T::Mat_xx A_d(double /*dt*/, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override { return T::Mat_xx::Identity(); }

  /** Get the Jacobian of the continuous state transition model with respect to the process noise.
   * @param dt Time step
   * @param x State (unused)
   * @return T::Mat_xv
   * @note Overriding DynamicModelLTV::G_d
   */
  T::Mat_xv G_d(double dt, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    T::Mat_xx I = T::Mat_xx::Identity();
    return 0.5 * dt * I;
  }

  /** Get the continuous time process noise covariance matrix.
   * @param dt Time step (unused)
   * @param x State (unused)
   * @return T::Mat_xx Process noise covariance
   * @note Overriding DynamicModelLTV::Q_d
   */
  T::Mat_vv Q_d(double /*dt*/ = 0.0, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override { return T::Mat_vv::Identity() * std_pos_ * std_pos_; }

private:
  double std_pos_;
};

/** (Nearly) Constant Velocity Model.
 * State x = [pos, vel], where pos and vel are `n_spatial_dim`-dimensional vectors
 * @tparam n_spatial_dim Number of spatial dimensions
 */
class ConstantVelocity : public interface::DynamicModelLTV<4, UNUSED, 2> {
  using Parent = interface::DynamicModelLTV<4, UNUSED, 2>;

public:
  static constexpr int N_DIM_x = Parent::N_DIM_x;
  static constexpr int N_DIM_u = Parent::N_DIM_u;
  static constexpr int N_DIM_v = Parent::N_DIM_v;

  static constexpr int N_SPATIAL_DIM = 2;
  static constexpr int N_STATES      = 2 * N_SPATIAL_DIM;

  using T = vortex::Types_xuv<N_DIM_x, N_DIM_u, N_DIM_v>;


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
   * @return T::Mat_xx
   * @note Overriding DynamicModelLTV::A_d
   */
  T::Mat_xx A_d(double dt, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_ss O = Mat_ss::Zero();
    T::Mat_xx A;
    // clang-format off
    A << I, I*dt,
         O, I;
    // clang-format on
    return A;
  }

  /** Get the Jacobian of the continuous state transition model with respect to the process noise.
   * @param dt Time step
   * @param x State (unused)
   * @return T::Mat_xv
   * @note Overriding DynamicModelLTV::G_d
   */
  T::Mat_xv G_d(double dt, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    T::Mat_xv G;
    // clang-format off
    G << 0.5*dt*dt*I,
                dt*I;
    // clang-format on
    return G;
  }

  /** Get the continuous time process noise covariance matrix.
   * @param dt Time step (unused)
   * @param x State (unused)
   * @return T::Mat_xx Process noise covariance
   * @note Overriding DynamicModelLTV::Q_d
   */
  T::Mat_vv Q_d(double /*dt*/ = 0.0, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override { return T::Mat_vv::Identity() * std_vel_ * std_vel_; }

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

  using T = vortex::Types_xv<N_STATES, N_DIM_v>;

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
   * @return T::Mat_xx
   * @note Overriding DynamicModelLTV::A_d
   */
  T::Mat_xx A_d(double dt, const T::Vec_x /*x*/ &) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_ss O = Mat_ss::Zero();
    T::Mat_xx A;
    // clang-format off
    A << I, I*dt, I*0.5*dt*dt,
         O, I   , I*dt       ,
         O, O   , I          ;
    // clang-format on
    return A;
  }

  T::Mat_xv G_d(double dt, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    Mat_ss I = Mat_ss::Identity();
    Mat_ss O = Mat_ss::Zero();
    T::Mat_xv G;
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
   * @return T::Mat_xx Process noise covariance
   * @note Overriding DynamicModelLTV::Q_d
   */
  T::Mat_vv Q_d(double /*dt*/ = 0.0, const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    T::Vec_v D;
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
  using T = vortex::Types_xv<N_DIM_x, N_DIM_v>;

  using ST = StateType;
  static constexpr std::array<ST, N_DIM_x> StateNames{ST::position, ST::position, ST::velocity, ST::velocity, ST::turn_rate};
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
   * @return T::Mat_xx
   * @note Overriding DynamicModelCTLTV::A_c
   */
  T::Mat_xx A_c(const T::Vec_x /*x*/ &x) const override
  {
    // clang-format off
    return T::Mat_xx{
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
   * return T::Mat_xv Process noise matrix
   * @note Overriding DynamicModelCTLTV::G_c
   */
  T::Mat_xv G_c(const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    // clang-format off
    return T::Mat_xv {
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
   * @return T::Mat_xx Process noise covariance
   * @note Overriding DynamicModelCTLTV::Q_c
   */
  T::Mat_vv Q_c(const T::Vec_x /*x*/ & = T::Vec_x::Zero()) const override
  {
    double var_vel  = std_vel_ * std_vel_;
    double var_turn = std_turn_ * std_turn_;

    return T::Vec_v{var_vel, var_vel, var_turn}.asDiagonal();
  }

private:
  double std_vel_;
  double std_turn_;
};

} // namespace models
} // namespace vortex