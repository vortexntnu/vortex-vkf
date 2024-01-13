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
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>

namespace vortex::filter {

template <typename SensModT, typename ImmModelT> class ImmFilter {
public:
  using SensModTPtr = std::shared_ptr<SensModT>;
  using SensModI    = typename SensModT::SensModI;

  static constexpr size_t N_MODELS = ImmModelT::N_MODELS;
  static constexpr int N_DIM_x     = SensModI::N_DIM_x;
  static constexpr int N_DIM_z     = SensModI::N_DIM_z;

  using Vec_n  = typename ImmModelT::Vec_n;
  using Mat_nn = typename ImmModelT::Mat_nn;

  using Vec_x = typename SensModI::Vec_x;
  using Vec_z = typename SensModI::Vec_z;

  using Mat_xz = typename SensModI::Mat_xz;
  using Mat_zz = typename SensModI::Mat_zz;

  using Gauss_x = typename SensModI::Gauss_x;
  using Gauss_z = typename SensModI::Gauss_z;

  using Vec_Gauss_x = std::vector<Gauss_x>;
  using Vec_Gauss_z = std::vector<Gauss_z>;

  using GaussMix_x = prob::GaussianMixture<N_DIM_x>;
  using GaussMix_z = prob::GaussianMixture<N_DIM_z>;

  ImmFilter()
  {
    // Check if the number of states in each dynamic model is the same as the number of states in the sensor model
    for (size_t n_dim_x : ImmModelT::get_n_dim_x()) {
      if (n_dim_x != N_DIM_x) {
        throw std::invalid_argument("Number of states in dynamic models does not match the number of states in the sensor model.");
      }
    }
  }

  /**
   * Calculates the mixing probabilities for the IMM filter, following step 1 in (6.4.1) in the book.
   *
   * @param model_weights The weights (mode probabilities) from the previous time step.
   * @param dt The time step.
   * @return The mixing probabilities. Each element is mixing_probs[s_{k-1}, s_k] = mu_{s_{k-1}|s_k} where s is the index of the model.
   */
  Mat_nn calculate_mixing_probs(const ImmModelT &imm_model, const Vec_n &model_weights, double dt)
  {
    Mat_nn pi_mat = imm_model.get_pi_mat_d(dt);

    // mu_{s_{k-1}|s_k} = pi_{s_{k-1}|s_k} * mu_{s_{k-1}|k-1}
    Mat_nn mixing_probs = pi_mat.cwiseProduct(model_weights.replicate(1, N_MODELS));

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
   * @return vector of moment-based predictions, i.e. update each model based on the state of all of the other models.
   */
  std::vector<Gauss_x> mixing(const std::vector<Gauss_x> &x_est_prevs, const Mat_nn &mixing_probs)
  {
    std::vector<Gauss_x> moment_based_preds;
    for (auto weights : mixing_probs.rowwise()) {
      GaussMix_x mixture(weights, x_est_prevs);
      moment_based_preds.push_back(mixture.reduce());
    }
    return moment_based_preds;
  }

  /**
   * @brief Calculate the Kalman filter outputs for each mode (6.36), following step 3 in (6.4.1) in the book
   * @param dt double Time step
   * @param moment_based_preds Moment-based predictions
   * @param z_meas Vec_z Measurement
   * @return Tuple of updated states, predicted states, predicted measurements
   */
  std::tuple<Vec_Gauss_x, Vec_Gauss_x, Vec_Gauss_z> mode_matched_filter(const ImmModelT &imm_model, const SensModTPtr &sensor_model,
                                                                        double dt, const std::vector<Gauss_x> &moment_based_preds,
                                                                        const Vec_z &z_meas)
  {
    return mode_matched_filter_impl(imm_model, sensor_model, dt, moment_based_preds, z_meas, std::make_index_sequence<ImmModelT::N_MODELS>{});
  }

  template <size_t i> std::tuple<Gauss_x, Gauss_x, Gauss_z> step_kalman_filter(const ImmModelT &imm_model, const SensModTPtr &sensor_model,
                                                                               double dt, const Gauss_x &x_est_prev, const Vec_z &z_meas)
  {
    using DynModI    = typename ImmModelT::template DynModI<i>;
    using DynModIPtr = typename DynModI::SharedPtr;

    filter::EKF_M<DynModI, SensModI> ekf;
    DynModIPtr dyn_model = imm_model.template get_model<i>();
    return ekf.step(dyn_model, sensor_model, dt, x_est_prev, z_meas);
  }

  /**
   * @brief Update the mode probabilites based on how well the predictions matched the measurements.
   * Using (6.37) from step 3 and (6.38) from step 4 in (6.4.1) in the book
   * @param dt double Time step
   * @param z_preds Mode-match filter outputs
   * @param z_meas Vec_z Measurement
   * @param prev_weigths Vec_n Weights
   * @return `Vec_n` Updated weights
   */
  Vec_n update_probabilities(const ImmModelT& imm_model, double dt, const std::vector<Gauss_z> &z_preds, const Vec_z &z_meas, const Vec_n &prev_weights)
  {
    Mat_nn pi_mat      = imm_model.get_pi_mat_d(dt);
    Vec_n weights_pred = pi_mat.transpose() * prev_weights;

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
   * @return Tuple of updated mixture, predicted mixture, predicted measurement mixture
   */
  GaussMix_x step(double dt, const GaussMix_x &x_est_prev, const Vec_z &z_meas)
  {
    Mat_nn mixing_probs                         = calculate_mixing_probs(x_est_prev.weights(), dt);
    std::vector<Gauss_x> moment_based_preds     = mixing(x_est_prev.gaussians(), mixing_probs);
    auto [x_est_upds, x_est_preds, z_est_preds] = mode_matched_filter(moment_based_preds, z_meas, dt);
    Vec_n weights_upd                           = update_probabilities(dt, z_est_preds, z_meas, x_est_prev.weights());

    return {weights_upd, x_est_upds};
  }

private:

  template <size_t... Is>
  std::tuple<Vec_Gauss_x, Vec_Gauss_x, Vec_Gauss_z>
  mode_matched_filter_impl(const ImmModelT &imm_model, const SensModTPtr &sensor_model,
                           double dt, const std::vector<Gauss_x> &moment_based_preds, const Vec_z &z_meas, std::index_sequence<Is...>)
  {

    // Calculate mode-matched filter outputs and save them in a vector of tuples
    std::vector<std::tuple<Gauss_x, Gauss_x, Gauss_z>> ekf_outs;
    ((ekf_outs.push_back(step_kalman_filter<Is>(imm_model, sensor_model, dt, moment_based_preds.at(Is), z_meas))), ...);

    // Convert vector of tuples to tuple of vectors
    std::vector<Gauss_x> x_est_upds;
    std::vector<Gauss_x> x_est_preds;
    std::vector<Gauss_z> z_est_preds;

    for (auto [x_est_upd, x_est_pred, z_est_pred] : ekf_outs) {
      x_est_upds.push_back(x_est_upd);
      x_est_preds.push_back(x_est_pred);
      z_est_preds.push_back(z_est_pred);
    }

    return {x_est_upds, x_est_preds, z_est_preds};
  }
};

} // namespace vortex::filter