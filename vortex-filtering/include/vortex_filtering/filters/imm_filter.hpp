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

#include <memory>
#include <tuple>

#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>

#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/uniform.hpp>

#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

namespace vortex::filter {

template <models::concepts::SensorModel SensModT, models::concepts::ImmModel ImmModelT> class ImmFilter {
public:
  using SensModTPtr = std::shared_ptr<SensModT>;
  using SensModI    = typename SensModT::SensModI;

  static constexpr size_t N_MODELS = ImmModelT::N_MODELS;

  static constexpr int N_DIM_z     = SensModI::N_DIM_z;

  using Vec_n  = typename ImmModelT::Vec_n;
  using Mat_nn = typename ImmModelT::Mat_nn;

  using Vec_x = typename SensModI::Vec_x;
  using Vec_z = typename SensModI::Vec_z;

  using Mat_xz = typename SensModI::Mat_xz;
  using Mat_zz = typename SensModI::Mat_zz;


  using Gauss_z = typename SensModI::Gauss_z;
  using GaussArr_z = std::array<Gauss_z, N_MODELS>;


  using GaussMix_z = prob::GaussianMixture<N_DIM_z>;

  template<size_t i> using DynModT    = typename ImmModelT::template DynModT<i>;
  template<size_t i> using DynModTPtr = typename ImmModelT::template DynModTPtr<i>;
  template<size_t i> using DynModI    = typename ImmModelT::template DynModI<i>;
  template<size_t i> using DynModIPtr = typename ImmModelT::template DynModIPtr<i>;

  template<size_t i> using Gauss_x    = typename ImmModelT::template Gauss_x<i>;
  template<size_t i> using GaussMix_x = typename ImmModelT::template GaussMix_x<i>;

  using GaussTuple_x = typename ImmModelT::GaussTuple_x;

  ImmFilter()
  {
    // Check if the number of states in each dynamic model is the same as the number of states in the sensor model
    // static_assert(ImmModelT::SAME_DIMS_x, "The number of states in each dynamic model must be the same as the number of states in the sensor model.");
  }

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
   * @return tuple of moment-based predictions, i.e. update each model based on the state of all of the other models.
   */
  static GaussTuple_x mixing(const GaussTuple_x &x_est_prevs, const Mat_nn &mixing_probs)
  {
    return mix_components(x_est_prevs, mixing_probs, std::make_index_sequence<N_MODELS>{});
  }

  /**
   * @brief Calculate moment-based approximation, following step 2 in (6.4.1) in the book
   * @param x_est_prev Gaussians from previous time step
   * @param mixing_probs Mixing probabilities
   * @return vector of moment-based predictions, i.e. update each model based on the state of all of the other models.
   */
  // static std::vector<Gauss_x> mixing(const std::vector<Gauss_x> &x_est_prevs, const Mat_nn &mixing_probs)
  // {
  //   std::vector<Gauss_x> moment_based_preds;
  //   for (const Vec_n &weights : mixing_probs.rowwise()) {
  //     GaussMix_x mixture(weights, x_est_prevs);
  //     moment_based_preds.push_back(mixture.reduce());
  //   }
  //   return moment_based_preds;
  // }

  /**
   * @brief Calculate the Kalman filter outputs for each mode (6.36), following step 3 in (6.4.1) in the book
   * @param imm_model The IMM model.
   * @param sensor_model The sensor model.
   * @param dt double Time step
   * @param moment_based_preds Moment-based predictions
   * @param z_meas Vec_z Measurement
   * @return Tuple of updated states, predicted states, predicted measurements
   */
  static std::tuple<GaussTuple_x, GaussTuple_x, GaussArr_z> mode_matched_filter(const ImmModelT &imm_model, const SensModTPtr &sensor_model, double dt,
                                                                               const GaussTuple_x &moment_based_preds, const Vec_z &z_meas)
  {
    return step_kalman_filters(imm_model, sensor_model, dt, moment_based_preds, z_meas, std::make_index_sequence<N_MODELS>{});
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
  static std::tuple<Gauss_x<i>, Gauss_x<i>, Gauss_z> step_kalman_filter(const DynModTPtr<i> &dyn_model, const SensModTPtr &sensor_model, double dt,
                                                                  const Gauss_x<i> &x_est_prev, const Vec_z &z_meas)
  {
    if constexpr (models::concepts::DynamicModelLTV<DynModT<i>> && models::concepts::SensorModelLTV<SensModT>) {
      using ImmSensMod = models::ImmSensorModelLTV<ImmModelT::N_DIM_x(i), SensModT>;
      using EKF = filter::EKF<DynModI<i>, ImmSensMod>;
      auto imm_sens_mod = std::make_shared<ImmSensMod>(sensor_model);
      return EKF::step(dyn_model, imm_sens_mod, dt, x_est_prev, z_meas);
    }
    else {
      using ImmSensMod = models::ImmSensorModel<ImmModelT::N_DIM_x(i), SensModT>;
      using UKF = filter::UKF<DynModI<i>, ImmSensMod>;
      auto imm_sens_mod = std::make_shared<ImmSensMod>(sensor_model);
      return UKF::step(dyn_model, imm_sens_mod, dt, x_est_prev, z_meas);
    }
  }

  /**
   * @brief Update the mode probabilites based on how well the predictions matched the measurements.
   * Using (6.37) from step 3 and (6.38) from step 4 in (6.4.1) in the book
   * @param transition_matrix The discrete time transition matrix for the Markov chain. (pi_mat_d from the imm model)
   * @param dt double Time step
   * @param z_preds Mode-match filter outputs
   * @param z_meas Vec_z Measurement
   * @param prev_weigths Vec_n Weights
   * @return `Vec_n` Updated weights
   */
  static Vec_n update_probabilities(const Mat_nn &transition_matrix, const std::vector<Gauss_z> &z_preds, const Vec_z &z_meas, const Vec_n &prev_weights)
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
   * @param dt Time step
   * @param x_est_prev Mixture from previous time step
   * @param z_meas Vec_z
   * @return Tuple of updated weights and predictions
   */
  static std::tuple<Vec_n, GaussTuple_x> step(const ImmModelT &imm_model, const SensModTPtr &sensor_model, double dt, const GaussTuple_x &x_est_prevs, const Vec_n &weights, const Vec_z &z_meas)
  {
    Mat_nn transition_matrix = imm_model.get_pi_mat_d(dt);

    Mat_nn mixing_probs                         = calculate_mixing_probs(transition_matrix, weights, dt);
    GaussTuple_x moment_based_preds             = mixing(x_est_prevs, mixing_probs);
    auto [x_est_upds, x_est_preds, z_est_preds] = mode_matched_filter(imm_model, sensor_model, moment_based_preds, z_meas, dt);
    Vec_n weights_upd                           = update_probabilities(transition_matrix, dt, z_est_preds, z_meas, weights);

    return {weights_upd, x_est_upds};
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
  static std::tuple<GaussTuple_x, GaussTuple_x, GaussArr_z> step_kalman_filters(const ImmModelT &imm_model, const SensModTPtr &sensor_model, double dt,
                                                                                const GaussTuple_x &moment_based_preds, const Vec_z &z_meas,
                                                                                std::index_sequence<Is...>)
  {

    // Calculate mode-matched filter outputs and save them in a tuple of tuples
    std::tuple<std::tuple<Gauss_x<Is>, Gauss_x<Is>, Gauss_z>...> ekf_outs;
    ((std::get<Is>(ekf_outs) = step_kalman_filter<Is>(imm_model.template get_model<Is>(), sensor_model, dt, std::get<Is>(moment_based_preds), z_meas)), ...);

    // Convert tuple of tuples to tuple of tuples
    GaussTuple_x x_est_upds;
    GaussTuple_x x_est_preds;
    GaussArr_z z_est_preds;

    ((std::get<Is>(x_est_upds)  = std::get<0>(std::get<Is>(ekf_outs))), ...);
    ((std::get<Is>(x_est_preds) = std::get<1>(std::get<Is>(ekf_outs))), ...);
    ((z_est_preds.at(Is)        = std::get<2>(std::get<Is>(ekf_outs))), ...);

    return {x_est_upds, x_est_preds, z_est_preds};
  }

  template <size_t... model_indices>
  static std::tuple<Gauss_x<model_indices>...> mix_components(const GaussTuple_x &x_est_prevs, const Mat_nn &mixing_probs, std::integer_sequence<size_t, model_indices...>)
  {
    return {mix_one_component<model_indices>(x_est_prevs, mixing_probs.col(model_indices))...};
  }

  template <size_t target_model_index>
  static Gauss_x<target_model_index> mix_one_component(const GaussTuple_x &x_est_prevs, const Vec_n &weights)
  {
    using GaussMix_x        = prob::GaussianMixture<ImmModelT::N_DIM_x(target_model_index)>;
    auto moment_based_preds = fit_models<target_model_index>(x_est_prevs, std::make_index_sequence<N_MODELS>{});
    return GaussMix_x(weights, moment_based_preds).reduce();
  }

  template <size_t target_model_index, size_t... model_indices>
  static std::array<Gauss_x<target_model_index>, N_MODELS> fit_models(const GaussTuple_x &x_est_prevs, std::integer_sequence<size_t, model_indices...>)
  {
    return {fit_one_model<target_model_index, model_indices>(x_est_prevs)...};
  }

  /**
   * @brief Fit the mixing_model in case it doesn't have the same dimensions or states as the target model 
   * 
   * @tparam target_model_index The model to fit to
   * @tparam mixing_model_index The model to fit
   * @param x_est_prevs 
   * @return Gauss_x<target_model_index> 
   */
  template <size_t target_model_index, size_t mixing_model_index>
  static Gauss_x<target_model_index> fit_one_model(const GaussTuple_x &x_est_prevs)
  {
      constexpr size_t N_DIM_x = ImmModelT::N_DIM_x(target_model_index);
      using Vec_x = Eigen::Vector<double, N_DIM_x>;
      using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
      // using Gauss_x = prob::Gauss<N_DIM_x>;
      using Uniform_x = prob::Uniform<N_DIM_x>;
      using MatchedVec = Eigen::Vector<bool, N_MODELS>;

      auto target_state_names = ImmModelT::template get_state_names<target_model_index>();
      auto mixing_state_names = ImmModelT::template get_state_names<mixing_model_index>();

      MatchedVec matching_states = Eigen::Map<MatchedVec>(matching_state_names(target_state_names, mixing_state_names).data());

      Vec_x x = std::get<mixing_model_index>(x_est_prevs).mean().cwiseProduct(matching_states.template cast<double>());
      Mat_xx P = std::get<mixing_model_index>(x_est_prevs).cov().cwiseProduct((matching_states * matching_states.transpose()).template cast<double>());

      for (size_t i = 0; i < N_DIM_x; i++) {
          if (!matching_states(i)) {
              Uniform_x initial_estimate{-Vec_x::Ones(), Vec_x::Ones()};
              x(i) = initial_estimate.mean()(i);
              P(i, i) = initial_estimate.cov()(i, i);
          }
      }

      return {x, P};
  }



    
};

} // namespace vortex::filter