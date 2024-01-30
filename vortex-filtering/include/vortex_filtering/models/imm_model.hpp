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

/**
 * @brief Container class for interacting multiple models.
 * @tparam DynModels Dynamic models to use.
 * @note The models must have a `DynModI` typedef specifying the base interface (e.g. `using DynModI = ...`).
 */
template <concepts::DynamicModel... DynModels> class ImmModel {
public:
  using DynModTuple    = std::tuple<DynModels...>;
  using DynModPtrTuple = std::tuple<std::shared_ptr<DynModels>...>;

  static constexpr size_t N_MODELS = sizeof...(DynModels);

  using Vec_n  = Eigen::Vector<double, N_MODELS>;
  using Mat_nn = Eigen::Matrix<double, N_MODELS, N_MODELS>;

  template <size_t i> using DynModI = typename std::tuple_element<i, DynModTuple>::type::DynModI; // Get the base interface of the i'th model
  template <size_t i> using Vec_x   = typename DynModI<i>::Vec_x;
  template <size_t i> using Vec_u   = typename DynModI<i>::Vec_u;
  template <size_t i> using Vec_v   = typename DynModI<i>::Vec_v;
  template <size_t i> using Mat_xx  = typename DynModI<i>::Mat_xx;
  template <size_t i> using Mat_vv  = typename DynModI<i>::Mat_vv;

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

  /**
   * @brief Get the number of dimensions for the state vector of each dynamic model
   * @return std::array<int, N_MODELS>
   */
  static constexpr std::array<int, N_MODELS> get_n_dim_x() { return {DynModels::DynModI::N_DIM_x...}; }

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