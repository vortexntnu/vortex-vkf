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
#include <ranges>
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

template <models::concepts::ImmModel ImmModelT, concepts::SensorModelWithDefinedSizes SensModT> class ImmFilter {
public:
  static constexpr size_t N_MODELS = ImmModelT::N_MODELS;

  static constexpr auto N_DIMS_x = ImmModelT::N_DIMS_x;
  static constexpr auto N_DIMS_u = ImmModelT::N_DIMS_u;
  static constexpr auto N_DIMS_v = ImmModelT::N_DIMS_v;
  static constexpr int N_DIM_z   = SensModT::N_DIM_z;
  static constexpr int N_DIM_w   = SensModT::N_DIM_w;

  template <size_t s_k> using T = Types_xz<N_DIMS_x.at(s_k), N_DIM_z>;

  template <size_t s_k> using DynModT = typename ImmModelT::template DynModT<s_k>;

  using Vec_n        = Eigen::Vector<double, N_MODELS>;
  using Arr_nb       = Eigen::Array<bool, N_MODELS, 1>;
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
    for (int s_k = 0; s_k < mixing_probs.cols(); s_k++) {
      mixing_probs.col(s_k) /= mixing_probs.col(s_k).sum();
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
  static GaussTuple_x mixing(const GaussTuple_x &x_est_prevs, const Mat_nn &mixing_probs, const StateMap &states_min_max = {})
  {
    return [&]<size_t... s_k>(std::index_sequence<s_k...>) -> GaussTuple_x {
      return {mix_one_component<s_k>(x_est_prevs, mixing_probs.col(s_k), states_min_max)...};
    }(std::make_index_sequence<N_MODELS>{});

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
    GaussTuple_x x_est_upds;
    GaussTuple_x x_est_preds;
    GaussArr_z z_est_preds;

    [&]<std::size_t... s_k>(std::index_sequence<s_k...>) {
      ((std::tie(std::get<s_k>(x_est_upds), std::get<s_k>(x_est_preds), z_est_preds.at(s_k)) =
            kalman_filter_step<s_k>(imm_model.template get_model<s_k>(), sensor_model, dt, std::get<s_k>(moment_based_preds), z_meas)),
       ...);
    }(std::make_index_sequence<N_MODELS>{});

    return {x_est_upds, x_est_preds, z_est_preds};
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
    for (size_t s_k = 0; const Gauss_z &z_pred : z_preds) {
      z_probs(s_k++) = z_pred.pdf(z_meas);
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
                                                                        const StateMap &states_min_max = {})
  {
    Mat_nn transition_matrix = imm_model.get_pi_mat_d(dt);

    Mat_nn mixing_probs                         = calculate_mixing_probs(transition_matrix, weights);
    GaussTuple_x moment_based_preds             = mixing(x_est_prevs, mixing_probs, states_min_max);
    auto [x_est_upds, x_est_preds, z_est_preds] = mode_matched_filter(imm_model, sensor_model, dt, moment_based_preds, z_meas);
    Vec_n weights_upd                           = update_probabilities(mixing_probs, z_est_preds, z_meas, weights);

    return {weights_upd, x_est_upds, x_est_preds, z_est_preds};
  }

  static std::tuple<GaussTuple_x, GaussArr_z> kalman_filter_predictions(const ImmModelT &imm_model, const SensModT &sensor_model, double dt,
                                                                        const GaussTuple_x &x_est_prevs)
  {
    GaussTuple_x x_est_preds;
    GaussArr_z z_est_preds;

    [&]<size_t... s_k>(std::index_sequence<s_k...>) {
      ((std::tie(std::get<s_k>(x_est_preds), z_est_preds.at(s_k)) =
            kalman_filter_predict<s_k>(imm_model.template get_model<s_k>(), sensor_model, dt, std::get<s_k>(x_est_prevs))),
       ...);
    }(std::make_index_sequence<N_MODELS>{});

    return {x_est_preds, z_est_preds};
  }

  static GaussTuple_x kalman_filter_updates(const ImmModelT &imm_model, const SensModT &sensor_model, double dt, const GaussTuple_x &x_est_preds,
                                            const GaussArr_z &z_est_preds, const Vec_z &z_meas, const Arr_nb &gated_measurements = Arr_nb::Ones())
  {
    GaussTuple_x x_est_upds;

    [&]<size_t... s_k>(std::index_sequence<s_k...>) {
      ((std::get<s_k>(x_est_upds) =
            gated_measurements(s_k)
                ? kalman_filter_update<s_k>(imm_model.template get_model<s_k>(), sensor_model, dt, std::get<s_k>(x_est_preds), z_est_preds.at(s_k), z_meas)
                : std::get<s_k>(x_est_preds)),
       ...);
    }(std::make_index_sequence<N_MODELS>{});

    return x_est_upds;
  }

  /**
   * @brief Calculate the Kalman filter outputs for one mode. If the model isn't LTV, use the ukf instead of the ekf.
   * @tparam s_k Index of model
   * @param imm_model The IMM model.
   * @param sensor_model The sensor model.
   * @param dt double Time step
   * @param x_est_prev Moment-based prediction
   * @param z_meas Vec_z Measurement
   * @return Tuple of updated state, predicted state, predicted measurement
   */
  template <size_t s_k>
  static std::tuple<typename T<s_k>::Gauss_x, typename T<s_k>::Gauss_x, Gauss_z>
  kalman_filter_step(const DynModT<s_k> &dyn_model, const SensModT &sensor_model, double dt, const T<s_k>::Gauss_x &x_est_prev, const Vec_z &z_meas)
  {
    if constexpr (concepts::DynamicModelLTVWithDefinedSizes<DynModT<s_k>> && concepts::SensorModelLTVWithDefinedSizes<SensModT>) {
      using ImmSensMod = models::ImmSensorModelLTV<N_DIMS_x.at(s_k), SensModT>;
      using EKF        = filter::EKF_t<N_DIMS_x.at(s_k), N_DIM_z, N_DIMS_u.at(s_k), N_DIMS_v.at(s_k), N_DIM_w>;
      ImmSensMod imm_sens_mod{sensor_model};
      return EKF::step(dyn_model, imm_sens_mod, dt, x_est_prev, z_meas);
    }
    else {
      using ImmSensMod = models::ImmSensorModel<N_DIMS_x.at(s_k), SensModT>;
      using UKF        = filter::UKF_t<N_DIMS_x.at(s_k), N_DIM_z, N_DIMS_u.at(s_k), N_DIMS_v.at(s_k), N_DIM_w>;
      ImmSensMod imm_sens_mod{sensor_model};
      return UKF::step(dyn_model, imm_sens_mod, dt, x_est_prev, z_meas);
    }
  }

  template <size_t s_k>
  static std::tuple<typename T<s_k>::Gauss_x, Gauss_z> kalman_filter_predict(const DynModT<s_k> &dyn_model, const SensModT &sensor_model, double dt,
                                                                             const T<s_k>::Gauss_x &x_est_prev)
  {
    if constexpr (concepts::DynamicModelLTVWithDefinedSizes<DynModT<s_k>> && concepts::SensorModelLTVWithDefinedSizes<SensModT>) {
      using ImmSensMod = models::ImmSensorModelLTV<N_DIMS_x.at(s_k), SensModT>;
      using EKF        = filter::EKF_t<N_DIMS_x.at(s_k), N_DIM_z, N_DIMS_u.at(s_k), N_DIMS_v.at(s_k), N_DIM_w>;
      ImmSensMod imm_sens_mod{sensor_model};
      return EKF::predict(dyn_model, imm_sens_mod, dt, x_est_prev);
    }
    else {
      using ImmSensMod = models::ImmSensorModel<N_DIMS_x.at(s_k), SensModT>;
      using UKF        = filter::UKF_t<N_DIMS_x.at(s_k), N_DIM_z, N_DIMS_u.at(s_k), N_DIMS_v.at(s_k), N_DIM_w>;
      ImmSensMod imm_sens_mod{sensor_model};
      return UKF::predict(dyn_model, imm_sens_mod, dt, x_est_prev);
    }
  }

  template <size_t s_k>
  static T<s_k>::Gauss_x kalman_filter_update(const DynModT<s_k> &dyn_model, const SensModT &sensor_model, double dt, const T<s_k>::Gauss_x &x_est_pred,
                                              const T<s_k>::Gauss_z &z_est_pred, const Vec_z &z_meas)
  {
    if constexpr (concepts::DynamicModelLTVWithDefinedSizes<DynModT<s_k>> && concepts::SensorModelLTVWithDefinedSizes<SensModT>) {
      using ImmSensMod = models::ImmSensorModelLTV<N_DIMS_x.at(s_k), SensModT>;
      using EKF        = filter::EKF_t<N_DIMS_x.at(s_k), N_DIM_z, N_DIMS_u.at(s_k), N_DIMS_v.at(s_k), N_DIM_w>;
      ImmSensMod imm_sens_mod{sensor_model};
      return EKF::update(imm_sens_mod, x_est_pred, z_est_pred, z_meas);
    }
    else {
      using ImmSensMod = models::ImmSensorModel<N_DIMS_x.at(s_k), SensModT>;
      using UKF        = filter::UKF_t<N_DIMS_x.at(s_k), N_DIM_z, N_DIMS_u.at(s_k), N_DIMS_v.at(s_k), N_DIM_w>;
      ImmSensMod imm_sens_mod{sensor_model};
      return UKF::update(dyn_model, imm_sens_mod, dt, x_est_pred, z_est_pred, z_meas);
    }
  }

  /** Helper function to mix one component (mode) of the IMM filter. It mixes all the components with the target model.
   * @tparam target_model_index The model to mix the other models into
   * @param x_est_prevs Gaussians from previous time step
   * @param weights Weights (column of the mixing_probs matrix corresponding to the target model index)
   * @param states_min_max The minimum and maximum value each state can take (optional, but can lead to better performance)
   * @return Gauss_x<target_model_index> The updated model after mixing it with the other models
   * @note This is the function that actually does the mixing of the models. It is called for each model in the IMM filter.
   */
  template <size_t target_model_index>
  static T<target_model_index>::Gauss_x mix_one_component(const GaussTuple_x &x_est_prevs, const Vec_n &weights, const StateMap &states_min_max = {})
  {
    auto moment_based_preds =
        [&]<size_t... mixing_model_indices>(std::index_sequence<mixing_model_indices...>) -> std::array<typename T<target_model_index>::Gauss_x, N_MODELS> {
      return {prepare_mixing_model<target_model_index, mixing_model_indices>(x_est_prevs, states_min_max)...};
    }(std::make_index_sequence<N_MODELS>{});

    return typename T<target_model_index>::GaussMix_x{weights, moment_based_preds}.reduce();
  }

  /**
   * @brief Fit the size and states of the mixing_model in case it doesn't have the same dimensions or states as the target model.
   * A map of the minimum and maximum values for each state (`states_min_max`) can be provided to initialize the missing states.
   * If the `states_min_max` map is not provided, the missing states are initialized with the mean and covariance from the target model.
   * @tparam target_model_index The model to mix the other models into
   * @tparam mixing_model_index The model to mix into the target model
   * @param x_est_prevs Gaussians from previous time step
   * @param states_min_max The minimum and maximum value each state can take (optional, but can lead to better filter performance)
   * @return Gauss_x<target_model_index> Modified `mixing_model` to fit the size and types of `target_model`
   * @note - If the sizes and state names of the mixing model and the target model are the same, the mixing model is returned as is.
   * @note - If the sizes are different, the mixing model is modified to fit the size of the target model and the possibly missing states
   * are initialized with the mean and covariance from the target model or with a uniform distribution if `states_min_max` is provided.
   */
  template <size_t target_model_index, size_t mixing_model_index>
  static T<target_model_index>::Gauss_x prepare_mixing_model(const GaussTuple_x &x_est_prevs, const StateMap &states_min_max = {})
  {
    if constexpr (target_model_index == mixing_model_index) {
      return std::get<mixing_model_index>(x_est_prevs);
    }

    constexpr size_t N_DIM_target = ImmModelT::N_DIM_x(target_model_index);
    constexpr size_t N_DIM_mixing = ImmModelT::N_DIM_x(mixing_model_index);
    constexpr size_t N_DIM_min    = std::min(N_DIM_target, N_DIM_mixing);

    using Vec_x    = Eigen::Vector<double, N_DIM_target>;
    using Mat_xx   = Eigen::Matrix<double, N_DIM_target, N_DIM_target>;
    using Uniform  = prob::Uniform<1>;
    using Vec_x_b  = Eigen::Vector<bool, N_DIM_target>;
    using Mat_xx_b = Eigen::Matrix<bool, N_DIM_target, N_DIM_target>;

    auto target_state = std::get<target_model_index>(x_est_prevs);
    auto mixing_state = std::get<mixing_model_index>(x_est_prevs);

    Vec_x x  = Vec_x::Zero();
    Mat_xx P = Mat_xx::Zero();

    x.template head<N_DIM_min>()                     = mixing_state.mean().template head<N_DIM_min>();
    P.template topLeftCorner<N_DIM_min, N_DIM_min>() = mixing_state.cov().template topLeftCorner<N_DIM_min, N_DIM_min>();

    constexpr auto target_state_names = std::get<target_model_index>(ImmModelT::ALL_STATE_NAMES);
    constexpr auto mixing_state_names = std::get<mixing_model_index>(ImmModelT::ALL_STATE_NAMES);
    constexpr auto matching_states    = models::matching_state_names(target_state_names, mixing_state_names);

    constexpr bool all_states_match = std::apply([](auto... b) { return (b && ...); }, matching_states);

    if constexpr (all_states_match) {
      return {x, P};
    }

    Vec_x_b matching_states_vec_b = Eigen::Map<Vec_x_b>(models::matching_state_names(target_state_names, mixing_state_names).data());
    Vec_x matching_states_vec     = matching_states_vec_b.template cast<double>();

    Mat_xx_b matching_states_mat_b = matching_states_vec_b * matching_states_vec_b.transpose();
    Mat_xx matching_states_mat     = matching_states_mat_b.template cast<double>();

    // Set the states and covariances that are not in the target model to 0
    x = x.cwiseProduct(matching_states_vec);
    P = P.cwiseProduct(matching_states_mat);

    for (size_t i = 0; i < N_DIM_target; i++) {
      if (matching_states_vec(i))
        continue;
      StateName state_name = target_state_names.at(i);
      if (!states_min_max.contains(state_name)) {
        x(i)    = target_state.mean()(i);
        P(i, i) = target_state.cov()(i, i);
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