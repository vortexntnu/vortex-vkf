/**
 * @file imm_filter.hpp
 * @author Eirik Kolås
 * @brief IMM filter based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-11-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <array>
#include <memory>
#include <tuple>
#include <vector>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/uniform.hpp>
#include <vortex_filtering/types/model_concepts.hpp>

namespace vortex::filter {

template <concepts::model::SensorModelWithDefinedSizes SensModT, models::concepts::ImmModel ImmModelT> class ImmFilter {
public:
  static constexpr size_t N_MODELS = ImmModelT::N_MODELS;

  static constexpr auto N_DIMS_x = ImmModelT::N_DIMS_x;
  static constexpr int N_DIM_z   = SensModT::N_DIM_z;

  template <size_t i> using T = Types_xz<N_DIMS_x(i), N_DIM_z>;

  template <size_t i> using DynModT = typename ImmModelT::template DynModT<i>;

  using Vec_n        = Eigen::Vector<double, N_MODELS>;
  using Mat_nn       = Eigen::Matrix<double, N_MODELS, N_MODELS>;
  using Vec_z        = typename T<0>::Vec_z;
  using Gauss_z      = typename T<0>::Gauss_z;
  using GaussTuple_x = typename ImmModelT::GaussTuple_x;
  using GaussArr_z   = std::array<Gauss_z, N_MODELS>;

  /// No need to instantiate this class. All methods are static.
  ImmFilter() = delete;

  /**
   * Calculates the mixing probabilities for the IMM filter, following step 1 in (6.4.1) in the book.
   * @param transition_matrix The discrete time transition matrix for the Markov chain. (pi_mat_d from the imm model)
   * @param model_weights The weights (mode probabilities) from the previous time step.
   * @return The mixing probabilities. Each element is mixing_probs[s_{k-1}, s_k] = mu_{s_{k-1}|s_k} where s is the index of the model.
   */
  static Mat_nn calculate_mixing_probs(const Mat_nn &transition_matrix, const Vec_n &model_weights)
  {

    // mu_{s_{k-1}|s_k} = pi_{s_{k-1}|s_k} * mu_{s_{k-1}|k-1}
    Mat_nn mixing_probs = transition_matrix.cwiseProduct(model_weights.replicate(1, N_MODELS));

    // Normalize
    for (int i = 0; i < mixing_probs.cols(); i++) {
      mixing_probs.col(i) /= mixing_probs.col(i).sum();
    }
    return mixing_probs;
  }

  /**
   * @brief Calculate moment-based approximation, following step 2 in (6.4.1) in the book
   * @param x_est_prev Gaussians from previous time step
   * @param mixing_probs Mixing probabilities
   * @param state_names The names of the states
   * @param states_min_max The minimum and maximum value each state can take (optional, but can lead to better performance)
   * @return tuple of moment-based predictions, i.e. update each model based on the state of all of the other models.
   * @note - If the sizes are different, the mixing model is modified to fit the size of the target model and the possibly missing states
   * are initialized with the mean and covariance from the target model or with a uniform distribution if `states_min_max` is provided.
   */
  static GaussTuple_x mixing(const GaussTuple_x &x_est_prevs, const Mat_nn &mixing_probs, const ImmModelT::StateNames &state_names,
                             const models::StateMap &states_min_max = {})
  {
    return mix_components(x_est_prevs, mixing_probs, state_names, states_min_max, std::make_index_sequence<N_MODELS>{});
  }

  /**
   * @brief Calculate the Kalman filter outputs for each mode (6.36), following step 3 in (6.4.1) in the book
   * @param imm_model The IMM model.
   * @param sensor_model The sensor model.
   * @param dt double Time step
   * @param moment_based_preds Moment-based predictions
   * @param z_meas Vec_z Measurement
   * @return Tuple of updated states, predicted states, predicted measurements
   */
  static std::tuple<GaussTuple_x, GaussTuple_x, GaussArr_z> mode_matched_filter(const ImmModelT &imm_model, const SensModT &sensor_model, double dt,
                                                                                const GaussTuple_x &moment_based_preds, const Vec_z &z_meas)
  {
    return step_kalman_filters(imm_model, sensor_model, dt, moment_based_preds, z_meas, std::make_index_sequence<N_MODELS>{});
  }

  /**
   * @brief Update the mode probabilites based on how well the predictions matched the measurements.
   * Using (6.37) from step 3 and (6.38) from step 4 in (6.4.1) in the book
   * @param transition_matrix The discrete time transition matrix for the Markov chain. (pi_mat_d from the imm model)
   * @param z_preds Mode-match filter outputs
   * @param z_meas Vec_z Measurement
   * @param prev_weigths Vec_n Weights
   * @return `Vec_n` Updated weights
   */
  static Vec_n update_probabilities(const Mat_nn &transition_matrix, const GaussArr_z &z_preds, const Vec_z &z_meas, const Vec_n &prev_weights)
  {
    Vec_n weights_pred = transition_matrix.transpose() * prev_weights;

    Vec_n z_probs;
    for (size_t i = 0; i < N_MODELS; i++) {
      z_probs(i) = z_preds.at(i).pdf(z_meas);
    }

    Vec_n weights_upd = z_probs.cwiseProduct(weights_pred);
    weights_upd /= weights_upd.sum();

    return weights_upd;
  }

  /**
   * @brief Perform one IMM filter step
   * @param imm_model The IMM model.
   * @param sensor_model The sensor model.
   * @param dt double Time step
   * @param x_est_prevs Gaussians from previous time step
   * @param weights Vec_n Weights
   * @param z_meas Vec_z Measurement
   * @param states_min_max The minimum and maximum value each state can take (optional, but can lead to better performance)
   */
  static std::tuple<Vec_n, GaussTuple_x, GaussTuple_x, GaussArr_z> step(const ImmModelT &imm_model, const SensModT &sensor_model, double dt,
                                                                        const GaussTuple_x &x_est_prevs, const Vec_z &z_meas, const Vec_n &weights,
                                                                        const models::StateMap &states_min_max = {})
  {
    Mat_nn transition_matrix = imm_model.get_pi_mat_d(dt);

    Mat_nn mixing_probs                         = calculate_mixing_probs(transition_matrix, weights);
    GaussTuple_x moment_based_preds             = mixing(x_est_prevs, mixing_probs, imm_model.get_all_state_names(), states_min_max);
    auto [x_est_upds, x_est_preds, z_est_preds] = mode_matched_filter(imm_model, sensor_model, dt, moment_based_preds, z_meas);
    Vec_n weights_upd                           = update_probabilities(mixing_probs, z_est_preds, z_meas, weights);

    return {weights_upd, x_est_upds, x_est_preds, z_est_preds};
  }

private:
  /**
   * @brief Calculate the Kalman filter outputs for each mode.
   * @tparam Is Indices of models
   * @param imm_model The IMM model.
   * @param sensor_model The sensor model.
   * @param dt double Time step
   * @param moment_based_preds Moment-based predictions
   * @param z_meas Vec_z Measurement
   * @return Tuple of updated states, predicted states, predicted measurements
   */
  template <size_t... Is>
  static std::tuple<GaussTuple_x, GaussTuple_x, GaussArr_z> step_kalman_filters(const ImmModelT &imm_model, const SensModT &sensor_model, double dt,
                                                                                const GaussTuple_x &moment_based_preds, const Vec_z &z_meas,
                                                                                std::index_sequence<Is...>)
  {

    // Calculate mode-matched filter outputs and save them in a tuple of tuples
    std::tuple<std::tuple<typename T<Is>::Gauss_x, typename T<Is>::Gauss_x, Gauss_z>...> ekf_outs;
    ((std::get<Is>(ekf_outs) = step_kalman_filter<Is>(imm_model.template get_model<Is>(), sensor_model, dt, std::get<Is>(moment_based_preds), z_meas)), ...);

    // Convert tuple of tuples to tuple of tuples
    GaussTuple_x x_est_upds;
    GaussTuple_x x_est_preds;
    GaussArr_z z_est_preds;

    ((std::get<Is>(x_est_upds) = std::get<0>(std::get<Is>(ekf_outs))), ...);
    ((std::get<Is>(x_est_preds) = std::get<1>(std::get<Is>(ekf_outs))), ...);
    ((z_est_preds.at(Is) = std::get<2>(std::get<Is>(ekf_outs))), ...);

    return {x_est_upds, x_est_preds, z_est_preds};
  }

  /**
   * @brief Calculate the Kalman filter outputs for one mode. If the model isn't LTV, use the ukf instead of the ekf.
   * @tparam i Index of model
   * @param imm_model The IMM model.
   * @param sensor_model The sensor model.
   * @param dt double Time step
   * @param x_est_prev Moment-based prediction
   * @param z_meas Vec_z Measurement
   * @return Tuple of updated state, predicted state, predicted measurement
   */
  template <size_t i>
  static std::tuple<typename T<i>::Gauss_x, typename T<i>::Gauss_x, Gauss_z> step_kalman_filter(const DynModT<i> &dyn_model, const SensModT &sensor_model,
                                                                                                double dt, const T<i>::Gauss_x &x_est_prev, const Vec_z &z_meas)
  {
    if constexpr (concepts::model::DynamicModelLTVWithDefinedSizes<DynModT<i>> && concepts::model::SensorModelLTVWithDefinedSizes<SensModT>) {
      using ImmSensMod  = models::ImmSensorModelLTV<ImmModelT::N_DIM_x(i), SensModT>;
      using EKF         = filter::EKF<DynModT<i>, ImmSensMod>;
      ImmSensMod imm_sens_mod{sensor_model};
      return EKF::step(dyn_model, imm_sens_mod, dt, x_est_prev, z_meas);
    }
    else {
      using ImmSensMod  = models::ImmSensorModel<ImmModelT::N_DIM_x(i), SensModT>;
      using UKF         = filter::UKF<DynModT<i>, ImmSensMod>;
      ImmSensMod imm_sens_mod{sensor_model};
      return UKF::step(dyn_model, imm_sens_mod, dt, x_est_prev, z_meas);
    }
  }

  /** Helper function to mix the components (modes) of the IMM filter
   * @tparam model_indices
   * @param x_est_prevs Gaussians from previous time step
   * @param mixing_probs Mixing probabilities
   * @param state_names Names of the states
   * @param states_min_max The minimum and maximum value each state can take
   * @return std::tuple<Gauss_x<model_indices>...>
   */
  template <size_t... model_indices>
  static std::tuple<typename T<model_indices>::Gauss_x...> mix_components(const GaussTuple_x &x_est_prevs, const Mat_nn &mixing_probs,
                                                                          const ImmModelT::StateNames &state_names, const models::StateMap &states_min_max,
                                                                          std::integer_sequence<size_t, model_indices...>)
  {
    return {mix_one_component<model_indices>(x_est_prevs, mixing_probs.col(model_indices), state_names, states_min_max)...};
  }

  /** Helper function to mix one component (mode) of the IMM filter. It mixes all the components with the target model.
   * @tparam target_model_index The model to mix the other models into
   * @param x_est_prevs Gaussians from previous time step
   * @param weights Weights (column of the mixing_probs matrix corresponding to the target model index)
   * @param state_names Names of the states
   * @param states_min_max The minimum and maximum value each state can take (optional, but can lead to better performance)
   * @return Gauss_x<target_model_index> The updated model after mixing it with the other models
   * @note This is the function that actually does the mixing of the models. It is called for each model in the IMM filter.
   */
  template <size_t target_model_index>
  static T<target_model_index>::Gauss_x mix_one_component(const GaussTuple_x &x_est_prevs, const Vec_n &weights, const ImmModelT::StateNames &state_names,
                                                          const models::StateMap &states_min_max = {})
  {
    constexpr size_t N_DIM_x = ImmModelT::N_DIM_x(target_model_index);
    using GaussMix_x         = prob::GaussianMixture<N_DIM_x>;
    auto moment_based_preds  = prepare_models<target_model_index>(x_est_prevs, state_names, states_min_max, std::make_index_sequence<N_MODELS>{});
    return GaussMix_x{weights, moment_based_preds}.reduce();
  }

  /** Helper function to prepare the models for mixing in case of mismatching dimensions or state names
   * @tparam target_model_index The model to mix the other models into
   * @tparam mixing_model_indices The models to mix into the target model
   * @param x_est_prevs Gaussians from previous time step
   * @param state_names Names of the states
   * @param states_min_max The minimum and maximum value each state can take
   * @return std::array<Gauss_x<target_model_index>, N_MODELS>
   */
  template <size_t target_model_index, size_t... mixing_model_indices>
  static std::array<typename T<target_model_index>::Gauss_x, N_MODELS> prepare_models(const GaussTuple_x &x_est_prevs, const ImmModelT::StateNames &state_names,
                                                                                      const models::StateMap &states_min_max,
                                                                                      std::integer_sequence<size_t, mixing_model_indices...>)
  {
    return {prepare_mixing_model<target_model_index, mixing_model_indices>(x_est_prevs, state_names, states_min_max)...};
  }

  /** Fit the size of the mixing_model in case it doesn't have the same dimensions or states as the target model.
   * @tparam target_model_index The model to mix the other models into
   * @tparam mixing_model_index The model to mix into the target model
   * @param x_est_prevs Gaussians from previous time step
   * @param state_names Names of the states
   * @param states_min_max The minimum and maximum value each state can take (optional, but can lead to better performance)
   * @return Gauss_x<target_model_index> Modified `mixing_model` to fit the size and types of `target_model`
   * @note - If the sizes and state names of the mixing model and the target model are the same, the mixing model is returned as is.
   * @note - If the sizes are different, the mixing model is modified to fit the size of the target model and the possibly missing states
   * are initialized with the mean and covariance from the target model or with a uniform distribution if `states_min_max` is provided.
   */
  template <size_t target_model_index, size_t mixing_model_index>
  static T<target_model_index>::Gauss_x prepare_mixing_model(const GaussTuple_x &x_est_prevs, const ImmModelT::StateNames &state_names,
                                                             const models::StateMap &states_min_max = {})
  {
    if constexpr (target_model_index == mixing_model_index) {
      return std::get<mixing_model_index>(x_est_prevs);
    }

    constexpr size_t N_DIM_target = ImmModelT::N_DIM_x(target_model_index);
    constexpr size_t N_DIM_mixing = ImmModelT::N_DIM_x(mixing_model_index);
    constexpr size_t N_DIM_min    = std::min(N_DIM_target, N_DIM_mixing);

    using ST       = models::StateType;
    using Vec_x    = Eigen::Vector<double, N_DIM_target>;
    using Mat_xx   = Eigen::Matrix<double, N_DIM_target, N_DIM_target>;
    using Uniform  = prob::Uniform<1>;
    using Vec_x_b  = Eigen::Vector<bool, N_DIM_target>;
    using Mat_xx_b = Eigen::Matrix<bool, N_DIM_target, N_DIM_target>;

    auto target_state_names = std::get<target_model_index>(state_names);
    auto mixing_state_names = std::get<mixing_model_index>(state_names);
    auto matching_states    = matching_state_names(target_state_names, mixing_state_names);

    bool all_states_match = std::apply([](auto... b) { return (b && ...); }, matching_states);

    if (all_states_match) {
      Vec_x x  = std::get<mixing_model_index>(x_est_prevs).mean().template head<N_DIM_target>();
      Mat_xx P = std::get<mixing_model_index>(x_est_prevs).cov().template topLeftCorner<N_DIM_target, N_DIM_target>();
      return {x, P};
    }

    Vec_x x                                          = Vec_x::Zero();
    Mat_xx P                                         = Mat_xx::Zero();
    x.template head<N_DIM_min>()                     = std::get<mixing_model_index>(x_est_prevs).mean().template head<N_DIM_min>();
    P.template topLeftCorner<N_DIM_min, N_DIM_min>() = std::get<mixing_model_index>(x_est_prevs).cov().template topLeftCorner<N_DIM_min, N_DIM_min>();

    Vec_x_b matching_states_vec_b = Eigen::Map<Vec_x_b>(matching_state_names(target_state_names, mixing_state_names).data());
    Vec_x matching_states_vec     = matching_states_vec_b.template cast<double>();

    Mat_xx_b matching_states_mat_b = matching_states_vec_b * matching_states_vec_b.transpose();
    Mat_xx matching_states_mat     = matching_states_mat_b.template cast<double>();

    x = x.cwiseProduct(matching_states_vec);
    P = P.cwiseProduct(matching_states_mat);

    for (size_t i = 0; i < N_DIM_target; i++) {
      if (matching_states_vec(i))
        continue;
      ST state_name = target_state_names.at(i);
      if (!states_min_max.contains(state_name)) {
        x(i)    = std::get<target_model_index>(x_est_prevs).mean()(i);
        P(i, i) = std::get<target_model_index>(x_est_prevs).cov()(i, i);
        continue;
      }
      double min = states_min_max.at(state_name).min;
      double max = states_min_max.at(state_name).max;
      Uniform initial_estimate{min, max};
      x(i)    = initial_estimate.mean();
      P(i, i) = initial_estimate.cov();
    }

    return {x, P};
  }
};

} // namespace vortex::filter