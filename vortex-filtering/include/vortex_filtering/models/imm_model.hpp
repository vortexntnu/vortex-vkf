/**
 * @file imm_model.hpp
 * @author Eirik Kolås
 * @brief Container class for interacting multiple models.
 * @version 0.1
 * @date 2023-11-02
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::models {

template <typename T, size_t N, size_t M> constexpr std::array<bool, N> matching_state_names(const std::array<T, N> &array1, const std::array<T, M> &array2)
{
  std::array<bool, N> matches{};
  for (std::size_t i = 0; i < N; ++i) {
    matches[i] = i < N && i < M && array1.at(i) == array2.at(i);
  }
  return matches;
}

// structs for the type of state
enum class StateType { none, position, velocity, acceleration, turn_rate };
struct StateMinMax {
  double min;
  double max;
};
using StateMap = std::map<StateType, StateMinMax>;

/**
 * @brief Container class for interacting multiple models.
 * @tparam DynModels Dynamic models to use.
 */
template <vortex::concepts::DynamicModelWithDefinedSizes... DynModels> class ImmModel {
public:
  static constexpr std::array N_DIMS_x = {DynModels::N_DIM_x...};
  static constexpr std::array N_DIMS_u = {DynModels::N_DIM_u...};
  static constexpr std::array N_DIMS_v = {DynModels::N_DIM_v...};

  static constexpr bool SAME_DIMS_x = (DynModels::N_DIM_x == ...);
  static constexpr bool MIN_DIM_x   = std::min(N_DIMS_x);
  static constexpr size_t N_MODELS = sizeof...(DynModels);

  using DynModTuple  = std::tuple<DynModels...>;
  using GaussTuple_x = std::tuple<typename Types_x<DynModels::N_DIM_x>::Gauss_x...>;
  using StateNames   = std::tuple<std::array<StateType, DynModels::N_DIM_x>...>;
  using Vec_n        = Eigen::Vector<double, N_MODELS>;
  using Mat_nn       = Eigen::Matrix<double, N_MODELS, N_MODELS>;

  template <size_t i> using DynModT = typename std::tuple_element<i, DynModTuple>::type;

  template <size_t i> using T = Types_xuv<N_DIMS_x[i], N_DIMS_u[i], N_DIMS_v[i]>;

  /**
   * @brief Construct a new ImmModel object
   * @tparam DynModels Dynamic models to use. The models must be linear-time-varying and have a `DynModI` typedef
   * specifying the base interface as the LTV model interface or it's derived classes
   * (e.g. `using DynModI = interface::DynamicModelLTV<...>`).
   * @param jump_matrix Markov jump chain matrix for the transition probabilities.
   * I.e. the probability of switching from model i to model j is `jump_matrix(i,j)`. Diagonal should be 0.
   * @param hold_times Expected holding time in seconds for each state. Parameter is the mean of an exponential distribution.
   * @param models_and_state_names Tuple of dynamic models and an std::array of their state names. The state names is of the vortex::models::StateType enum.
   * @note - The jump matrix specifies the probability of switching to a model WHEN a switch occurs.
   * @note - The holding times specifies HOW LONG a state is expected to be held between switches.
   * @note - In order to change the properties of a model, you must get the model using `get_model<i>()`
   */
  ImmModel(Mat_nn jump_matrix, Vec_n hold_times, std::tuple<DynModels, std::array<StateType, DynModels::N_DIM_x>>... models_and_state_names)
      : ImmModel(jump_matrix, hold_times, std::get<0>(models_and_state_names)..., {std::get<1>(models_and_state_names)...})
  {
  }

  /**
   * @brief Construct a new ImmModel object
   * @tparam DynModels Dynamic models to use. The models must be linear-time-varying and have a `DynModI` typedef
   * specifying the base interface as the LTV model interface or it's derived classes
   * (e.g. `using DynModI = interface::DynamicModelLTV<...>`).
   * @param jump_matrix Markov jump chain matrix for the transition probabilities.
   * I.e. the probability of switching from model i to model j is `jump_matrix(i,j)`. Diagonal should be 0.
   * @param hold_times Expected holding time in seconds for each state. Parameter is the mean of an exponential distribution.
   * @param models Tuple of dynamic models
   * @param state_names Tuple of std::array of state names for each model
   * @note - The jump matrix specifies the probability of switching to a model WHEN a switch occurs.
   * @note - The holding times specifies HOW LONG a state is expected to be held between switches.
   * @note - In order to change the properties of a model, you must get the model using `get_model<i>()`
   */
  ImmModel(Mat_nn jump_matrix, Vec_n hold_times, DynModels... models, StateNames state_names)
      : models_(models...)
      , jump_matrix_(jump_matrix)
      , hold_times_(hold_times)
      , state_names_(state_names)
  {
    if (!jump_matrix_.diagonal().isZero()) {
      throw std::invalid_argument("Jump matrix diagonal should be zero");
    }

    // Check that the jump matrix is valid
    for (int i = 0; i < jump_matrix_.rows(); i++) {
      if (jump_matrix_.row(i).sum() != 1.0) {
        throw std::invalid_argument("Jump matrix row " + std::to_string(i) + " should sum to 1");
      }
    }
  }

  /**
   * @brief Compute the continuous-time transition matrix
   * @return Matrix Continuous-time transition matrix
   * @note See https://en.wikipedia.org/wiki/Continuous-time_Markov_chain.
   */
  Mat_nn get_pi_mat_c() const
  {
    Vec_n hold_rates = hold_times_.cwiseInverse();
    Mat_nn pi_mat_c  = hold_rates.replicate(1, N_MODELS);

    // Multiply the jump matrix with the hold times
    pi_mat_c = pi_mat_c.cwiseProduct(jump_matrix_);

    // Each row should sum to zero
    pi_mat_c -= hold_rates.asDiagonal();

    return pi_mat_c;
  }

  /**
   * @brief Compute the discrete-time transition matrix
   * @return Matrix Discrete-time transition matrix
   */
  Mat_nn get_pi_mat_d(double dt) const
  {
    // Cache the pi matrix for the same time step as it is likely to be reused and is expensive to compute
    static double prev_dt  = dt;
    static Mat_nn pi_mat_d = (get_pi_mat_c() * dt).exp();
    if (dt != prev_dt) {
      pi_mat_d = (get_pi_mat_c() * dt).exp();
      prev_dt  = dt;
    }
    return pi_mat_d;
  }

  /**
   * @brief Get the dynamic models
   * @return Reference to tuple of dynamic models
   */
  const DynModTuple &get_models() const { return models_; }

  /**
   * @brief Get the dynamic models (non-const)
   * @return Reference to tuple of dynamic models
   */
  DynModTuple &get_models() { return models_; }

  /**
   * @brief Get specific dynamic model
   * @tparam i Index of model
   * @return DynModT<i> model reference
   */
  template <size_t i> const DynModT<i> &get_model() const { return std::get<i>(models_); }

  /**
   * @brief Get specific dynamic model (non-const)
   * @tparam i Index of model
   * @return DynModT<i> model reference
   */
  template <size_t i> DynModT<i> &get_model() { return std::get<i>(models_); }

  /**
   * @brief f_d of specific dynamic model
   * @tparam i Index of model
   * @param dt Time step
   * @param x State
   * @param u Input (optional)
   * @param v Noise (optional)
   * @return Vec_x
   */
  template <size_t i>
  T<i>::Vec_x f_d(double dt, const T<i>::Vec_x &x, const T<i>::Vec_u &u = T<i>::Vec_u::Zero(), const T<i>::Vec_v &v = T<i>::Vec_v::Zero()) const
  {
    return get_model<i>().f_d(dt, x, u, v);
  }

  /**
   * @brief Q_d of specific dynamic model
   * @tparam i Index of model
   * @param dt Time step
   * @param x State
   * @return Mat_vv
   */
  template <size_t i> T<i>::Mat_vv Q_d(double dt, const T<i>::Vec_x &x) const { return get_model<i>().Q_d(dt, x); }

  static constexpr int N_DIM_x(size_t model_index) { return N_DIMS_x.at(model_index); }

  StateNames get_all_state_names() const { return state_names_; }

  template <size_t model_index> std::array<StateType, N_DIM_x(model_index)> get_state_names() { return std::get<model_index>(state_names_); }

  template <size_t model_index> StateType get_state_name(size_t i) { return get_state_names<model_index>().at(i); }

private:
  DynModTuple models_;
  Mat_nn jump_matrix_;
  Vec_n hold_times_;
  StateNames state_names_;
};

/**
 * @brief Class for resizing the state vector of a sensor model to fit with multiple dynamic models.
 *
 */
template <size_t n_dim_a, vortex::concepts::SensorModelWithDefinedSizes SensModT> class ImmSensorModel {
public:
  static constexpr int N_DIM_x_real = SensModT::N_DIM_x;
  static constexpr int N_DIM_z      = SensModT::N_DIM_z;
  static constexpr int N_DIM_w      = SensModT::N_DIM_w;
  static constexpr int N_DIM_a = (int)n_dim_a;
  static constexpr int N_DIM_x      = N_DIM_a; // For the consept to accept the state dimension of the sensor model (TODO: fix this in the future)

  using T = Types_xzwa<N_DIM_x_real, N_DIM_z, N_DIM_w, N_DIM_a>;

  ImmSensorModel(SensModT sensor_model)
      : sensor_model_(sensor_model)
  {
    static_assert(N_DIM_a >= SensModT::SensModI::N_DIM_x, "N_DIM_a must be greater than or equal to the state dimension of the sensor model");
  }

  T::Vec_z h(const T::Vec_a &x, const T::Vec_w &w) const { return sensor_model_.h(x.template head<N_DIM_x_real>(), w); }

  T::Mat_ww R(const T::Vec_x &x) const { return sensor_model_.R(x.template head<N_DIM_x_real>()); }

private:
  SensModT sensor_model_;
};

template <size_t n_dim_a, vortex::concepts::SensorModelLTVWithDefinedSizes SensModT> class ImmSensorModelLTV {
public:
  static constexpr int N_DIM_x_real = SensModT::N_DIM_x;
  static constexpr int N_DIM_z      = SensModT::N_DIM_z;
  static constexpr int N_DIM_w      = SensModT::N_DIM_w;
  static constexpr int N_DIM_a = (int)n_dim_a;
  static constexpr int N_DIM_x      = N_DIM_a; // For the consept to accept the state dimension of the sensor model (TODO: fix this in the future)

  using T = Types_xzwa<N_DIM_x_real, N_DIM_z, N_DIM_w, N_DIM_a>;

  ImmSensorModelLTV(SensModT sensor_model)
      : sensor_model_(sensor_model)
  {
    static_assert(N_DIM_a >= N_DIM_x, "N_DIM_a must be greater than or equal to the state dimension of the sensor model");
  }

  T::Vec_z h(const T::Vec_a &x, const T::Vec_w &w) const { return sensor_model_.h(x.template head<N_DIM_x_real>(), w); }

  T::Mat_za C(const T::Vec_a &x) const
  {
    typename T::Mat_za C_a                = T::Mat_za::Zero();
    C_a.template leftCols<N_DIM_x_real>() = sensor_model_.C(x.template head<N_DIM_x_real>());
    return C_a;
  }

  T::Mat_zw H(const T::Vec_a &x) const { return sensor_model_.H(x.template head<N_DIM_x_real>()); }

  T::Mat_ww R(const T::Vec_a &x) const { return sensor_model_.R(x.template head<N_DIM_x_real>()); }

  T::Gauss_z pred_from_est(const T::Gauss_a &x_est) const
  {
    typename T::Vec_x mean = x_est.mean().template head<N_DIM_x_real>();
    typename T::Mat_xx cov = x_est.cov().template topLeftCorner<N_DIM_x_real, N_DIM_x_real>();
    return sensor_model_.pred_from_est({mean, cov});
  }

  T::Gauss_z pred_from_state(const T::Vec_a &x) const { return sensor_model_.pred_from_state(x.template head<N_DIM_x_real>()); }

private:
  SensModT sensor_model_;
};

namespace concepts {
template <typename T>
concept ImmModel = requires {
  typename T::DynModTuple;
};

} // namespace concepts

} // namespace vortex::models