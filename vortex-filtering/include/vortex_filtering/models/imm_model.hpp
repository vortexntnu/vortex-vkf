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
#include <memory>
#include <tuple>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>

namespace vortex::models {
// namespace models {

namespace concepts {
  template <typename T>
  concept ImmCompatibleModel = requires {
    concepts::DynamicModel<T>; // Is a dynamic model
    typename T::StateNames; // Has a list of named states
    T::StateNames::N_STATES == T::DynModI::N_DIM_x;
  };
}

enum class StateType { none, position, velocity, acceleration, turn_rate };

template <StateType... state_types> struct SemanticState 
{
  constexpr SemanticState() = default;
  constexpr SemanticState(StateType...) {}

  static constexpr size_t N_STATES = sizeof...(state_types);
  static constexpr std::array<StateType, N_STATES> TYPES = {state_types...};
};

/**
 * @brief Container class for interacting multiple models.
 * @tparam DynModels Dynamic models to use.
 */
template <concepts::ImmCompatibleModel... DynModels> class ImmModel {
public:
  using DynModTuple    = std::tuple<DynModels...>;
  using DynModPtrTuple = std::tuple<std::shared_ptr<DynModels>...>;

  using GaussTuple_x = std::tuple<typename DynModels::DynModI::Gauss_x...>;

  static constexpr size_t N_MODELS = sizeof...(DynModels);

  static constexpr bool SAME_DIMS_x = (DynModels::DynModI::N_DIM_x == ...);

  using Vec_n  = Eigen::Vector<double, N_MODELS>;
  using Mat_nn = Eigen::Matrix<double, N_MODELS, N_MODELS>;

  template <size_t i> using DynModI = typename std::tuple_element<i, DynModTuple>::type::DynModI; // Get the base interface of the i'th model
  template <size_t i> using DynModIPtr = typename DynModI<i>::SharedPtr;
  template <size_t i> using Vec_x   = typename DynModI<i>::Vec_x;
  template <size_t i> using Vec_u   = typename DynModI<i>::Vec_u;
  template <size_t i> using Vec_v   = typename DynModI<i>::Vec_v;
  template <size_t i> using Mat_xx  = typename DynModI<i>::Mat_xx;
  template <size_t i> using Mat_vv  = typename DynModI<i>::Mat_vv;
  template <size_t i> using Gauss_x = typename DynModI<i>::Gauss_x;
  template <size_t i> using GaussMix_x = typename DynModI<i>::GaussMix_x;

  template <size_t i> using DynModT    = typename std::tuple_element<i, DynModTuple>::type;
  template <size_t i> using DynModTPtr = typename std::shared_ptr<DynModT<i>>;

  /**
   * @brief Construct a new Imm Model object
   * @tparam DynModels Dynamic models to use. The models must be linear-time-varying and have a `DynModI` typedef
   * specifying the base interface as the LTV model interface or it's derived classes
   * (e.g. `using DynModI = interface::DynamicModelLTV<...>`).
   * @param jump_matrix Markov jump chain matrix for the transition probabilities.
   * I.e. the probability of switching from model i to model j is `jump_matrix(i,j)`. Diagonal should be 0.
   * @param hold_times Expected holding time for each state. Parameter is the mean of an exponential distribution.
   * @param models Models to use. The models must have a DynModI typedef specifying the base interface.
   * @note - The jump matrix specifies the probability of switching to a model WHEN a switch occurs.
   * @note - The holding times specifies HOW LONG a state is expected to be held between switches.
   * @note - In order to change the properties of a model, you must get the model using `get_model<i>()`
   */
  ImmModel(Mat_nn jump_matrix, Vec_n hold_times, DynModels... models)
      : models_(std::make_shared<DynModels>(models)...), jump_matrix_(jump_matrix), hold_times_(hold_times)
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
    Mat_nn pi_mat_c = hold_times_.replicate(1, N_MODELS);

    // Multiply the jump matrix with the hold times
    pi_mat_c = pi_mat_c.cwiseProduct(jump_matrix_);

    // Each row should sum to zero
    pi_mat_c -= hold_times_.asDiagonal();

    return pi_mat_c;
  }

  /**
   * @brief Compute the discrete-time transition matrix
   * @return Matrix Discrete-time transition matrix
   */
  Mat_nn get_pi_mat_d(double dt) const { return (get_pi_mat_c() * dt).exp(); }

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


  static constexpr int N_DIM_x(size_t model_index) { return std::array<int, N_MODELS>{DynModels::DynModI::N_DIM_x...}.at(model_index); }


  template <size_t model_index> static constexpr std::array<StateType, N_DIM_x(model_index)> get_state_names() { return typename DynModT<model_index>::template StateNames::Types; }

  template <size_t model_index_1, size_t model_index_2>
  static constexpr bool same_state_name(size_t state_index)
  {
      size_t n_dim_x_1 = size_t(N_DIM_x(model_index_1));
      size_t n_dim_x_2 = size_t(N_DIM_x(model_index_2));

      // Return false if state_index is out of bounds for either model
      if (state_index >= n_dim_x_1 || state_index >= n_dim_x_2) {
          return false;
      }

      auto state_name_1 = get_state_names<model_index_1>().at(state_index);
      auto state_name_2 = get_state_names<model_index_2>().at(state_index);

      return state_name_1 == state_name_2;
  }

  template <size_t model_index_1, size_t model_index_2>
  static constexpr std::array<bool, model_index_1> MATCHING_STATE_NAMES()
  {
      std::array<bool, model_index_1> same_names{};

      for (size_t i = 0; i < model_index_1; i++) {
          same_names.at(i) = same_state_name<model_index_1, model_index_2>(i);
      }

      return same_names;
  }


private:
  DynModPtrTuple models_;
  const Mat_nn jump_matrix_;
  const Vec_n hold_times_;
};

namespace concepts {
template <typename T>
concept ImmModel = requires {
  typename T::DynModTuple;
  typename T::DynModPtrTuple;
};

} // namespace concepts

} // namespace vortex::models