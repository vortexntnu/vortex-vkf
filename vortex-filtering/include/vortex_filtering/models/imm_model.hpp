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
template <concepts::DynamicModel... DynModels> class ImmModel {
public:
  using DynModTuple    = std::tuple<DynModels...>;
  using DynModPtrTuple = std::tuple<std::shared_ptr<DynModels>...>;

  using GaussTuple_x = std::tuple<typename DynModels::DynModI::Gauss_x...>;
  using StateNames   = std::tuple<std::array<StateType, DynModels::DynModI::N_DIM_x>...>;

  static constexpr size_t N_MODELS = sizeof...(DynModels);

  static constexpr bool SAME_DIMS_x = (DynModels::DynModI::N_DIM_x == ...);

  using Vec_n  = Eigen::Vector<double, N_MODELS>;
  using Mat_nn = Eigen::Matrix<double, N_MODELS, N_MODELS>;

  template <size_t i> using DynModI    = typename std::tuple_element<i, DynModTuple>::type::DynModI; // Get the base interface of the i'th model
  template <size_t i> using DynModIPtr = typename DynModI<i>::SharedPtr;
  template <size_t i> using Vec_x      = typename DynModI<i>::Vec_x;
  template <size_t i> using Vec_u      = typename DynModI<i>::Vec_u;
  template <size_t i> using Vec_v      = typename DynModI<i>::Vec_v;
  template <size_t i> using Mat_xx     = typename DynModI<i>::Mat_xx;
  template <size_t i> using Mat_vv     = typename DynModI<i>::Mat_vv;
  template <size_t i> using Gauss_x    = typename DynModI<i>::Gauss_x;
  template <size_t i> using GaussMix_x = typename DynModI<i>::GaussMix_x;

  template <size_t i> using DynModT    = typename std::tuple_element<i, DynModTuple>::type;
  template <size_t i> using DynModTPtr = typename std::shared_ptr<DynModT<i>>;

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
  ImmModel(Mat_nn jump_matrix, Vec_n hold_times, std::tuple<DynModels, std::array<StateType, DynModels::DynModI::N_DIM_x>>... models_and_state_names)
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
      : models_(std::make_shared<DynModels>(models)...)
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
   * @return Reference to tuple of shared pointers to dynamic models
   */
  const DynModPtrTuple &get_models() const { return models_; }

  /**
   * @brief Get specific dynamic model
   * @tparam i Index of model
   * @return ModelT<i> shared pointer to model
   */
  template <size_t i> const DynModTPtr<i> &get_model() const { return std::get<i>(models_); }

  /**
   * @brief Get specific dynamic model (non-const)
   * @tparam i Index of model
   * @return ModelT<i> shared pointer to model
   */
  template <size_t i> DynModTPtr<i> get_model() { return std::get<i>(models_); }

  /**
   * @brief f_d of specific dynamic model
   * @tparam i Index of model
   * @param dt Time step
   * @param x State
   * @param u Input (optional)
   * @param v Noise (optional)
   * @return Vec_x
   */
  template <size_t i> Vec_x<i> f_d(double dt, const Vec_x<i> &x, const Vec_u<i> &u = Vec_u<i>::Zero(), const Vec_v<i> &v = Vec_v<i>::Zero()) const
  {
    return get_model<i>()->f_d(dt, x, u, v);
  }

  /**
   * @brief Q_d of specific dynamic model
   * @tparam i Index of model
   * @param dt Time step
   * @param x State
   * @return Mat_vv
   */
  template <size_t i> Mat_vv<i> Q_d(double dt, const Vec_x<i> &x) const { return get_model<i>()->Q_d(dt, x); }

  static constexpr std::array<int, N_MODELS> N_DIMS_x() { return std::array<int, N_MODELS>{DynModels::DynModI::N_DIM_x...}; }

  static constexpr int N_DIM_x(size_t model_index) { return N_DIMS_x().at(model_index); }

  StateNames get_all_state_names() const { return state_names_; }

  template <size_t model_index> std::array<StateType, N_DIM_x(model_index)> get_state_names() { return std::get<model_index>(state_names_); }

  template <size_t model_index> StateType get_state_name(size_t i) { return get_state_names<model_index>().at(i); }

private:
  DynModPtrTuple models_;
  Mat_nn jump_matrix_;
  Vec_n hold_times_;
  StateNames state_names_;
};

/**
 * @brief Class for resizing the state vector of a sensor model to fit with multiple dynamic models.
 *
 */
template <size_t n_dim_a, concepts::SensorModel SensModT> class ImmSensorModel {
public:
  using SensModI = typename SensModT::SensModI;

  static constexpr int N_DIM_x = SensModI::N_DIM_x;
  static constexpr int N_DIM_z = SensModI::N_DIM_z;
  static constexpr int N_DIM_a = (int)n_dim_a;

  using Vec_x = typename SensModI::Vec_x;
  using Vec_z = typename SensModI::Vec_z;
  using Vec_w = typename SensModI::Vec_w;
  using Vec_a = Eigen::Vector<double, N_DIM_a>;

  using Mat_ww = typename SensModI::Mat_ww;
  using Mat_aa = Eigen::Matrix<double, N_DIM_a, N_DIM_a>;

  ImmSensorModel(SensModT sensor_model)
      : sensor_model_(sensor_model)
  {
    static_assert(N_DIM_a >= SensModT::SensModI::N_DIM_x, "N_DIM_a must be greater than or equal to the state dimension of the sensor model");
  }

  Vec_z h(const Vec_a &x, const Vec_w &w) const { return sensor_model_->h(x.template head<N_DIM_x>(), w); }

  Mat_ww R(const Vec_x &x) const { return sensor_model_->R(x.template head<N_DIM_x>()); }

private:
  SensModT sensor_model_;
};

template <size_t n_dim_a, concepts::SensorModelLTV SensModT> class ImmSensorModelLTV {
public:
  using SensModI    = typename SensModT::SensModI;
  using SensModTPtr = typename std::shared_ptr<SensModT>;

  static constexpr int N_DIM_x = SensModI::N_DIM_x;
  static constexpr int N_DIM_z = SensModI::N_DIM_z;
  static constexpr int N_DIM_a = (int)n_dim_a;

  using Vec_x = typename SensModI::Vec_x;
  using Vec_z = typename SensModI::Vec_z;
  using Vec_w = typename SensModI::Vec_w;
  using Vec_a = Eigen::Vector<double, N_DIM_a>;

  using Mat_xx   = typename SensModI::Mat_xx;
  using Mat_ww   = typename SensModI::Mat_ww;
  using Mat_aa   = Eigen::Matrix<double, N_DIM_a, N_DIM_a>;
  using Mat_za   = Eigen::Matrix<double, N_DIM_z, N_DIM_a>;
  using Mat_zw   = typename SensModI::Mat_zw;
  using Mat_zamx = Eigen::Matrix<double, N_DIM_z, N_DIM_a - N_DIM_x>; // 'z' by 'a-x' matrix

  using Gauss_z = typename SensModI::Gauss_z;
  using Gauss_a = typename prob::MultiVarGauss<N_DIM_a>;

  ImmSensorModelLTV(SensModTPtr sensor_model)
      : sensor_model_(sensor_model)
  {
    static_assert(N_DIM_a >= N_DIM_x, "N_DIM_a must be greater than or equal to the state dimension of the sensor model");
  }

  Vec_z h(const Vec_a &x, const Vec_w &w) const { return sensor_model_->h(x.template head<N_DIM_x>(), w); }

  Mat_za C(const Vec_a &x) const
  {
    Mat_za C_a;
    C_a << sensor_model_->C(x.template head<N_DIM_x>()), Mat_zamx::Zero();
    return C_a;
  }

  Mat_zw H(const Vec_a &x) const { return sensor_model_->H(x.template head<N_DIM_x>()); }

  Mat_ww R(const Vec_a &x) const { return sensor_model_->R(x.template head<N_DIM_x>()); }

  Gauss_z pred_from_est(const Gauss_a &x_est) const
  {
    Vec_x mean = x_est.mean().template head<N_DIM_x>();
    Mat_xx cov = x_est.cov().template topLeftCorner<N_DIM_x, N_DIM_x>();
    return sensor_model_->pred_from_est({mean, cov});
  }

  Gauss_z pred_from_state(const Vec_a &x) const { return sensor_model_->pred_from_state(x.template head<N_DIM_x>()); }

private:
  SensModTPtr sensor_model_;
};

namespace concepts {
template <typename T>
concept ImmModel = requires {
  typename T::DynModTuple;
  typename T::DynModPtrTuple;
};

} // namespace concepts

} // namespace vortex::models