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

namespace vortex {
namespace filter {

template <class ImmModelT, class SensModT> class ImmFilter {
public:
  using SensModTPtr = std::shared_ptr<SensModT>;
  using SensModI    = typename SensModT::SensModI;

  static constexpr int N_MODELS = ImmModelT::N_MODELS;
  static constexpr int N_DIM_x  = SensModI::N_DIM_x;
  static constexpr int N_DIM_z  = SensModI::N_DIM_z;

  using Vec_n        = typename ImmModelT::Vec_n;
  using Mat_nn       = typename ImmModelT::Mat_nn;
  using Gauss_xTuple = typename ImmModelT::Gauss_xTuple;

  using Vec_x = typename SensModI::Vec_x;
  using Vec_z = typename SensModI::Vec_z;

  using Mat_xz = typename SensModI::Mat_xz;
  using Mat_zz = typename SensModI::Mat_zz;

  using Gauss_x = typename SensModI::Gauss_x;
  using Gauss_z = typename SensModI::Gauss_z;

  using GaussMix_x = prob::GaussianMixture<N_DIM_x>;
  using GaussMix_z = prob::GaussianMixture<N_DIM_z>;

  ImmFilter(ImmModelT imm_model, SensModTPtr sensor_model) : imm_model_(std::move(imm_model)), sensor_model_(std::move(sensor_model))
  {
    // Check if the number of states in each dynamic model is the same as the number of states in the sensor model
    std::apply(
        [](auto &&...n_dim_x) {
          if (!((N_DIM_x == n_dim_x) && ...)) {
            throw std::runtime_error("Number of states in dynamic models does not match the number of states in the sensor model.");
          }
        },
        ImmModelT::get_n_dim_x());
  }

  /**
   * Calculates the mixing probabilities for the IMM filter, following step 1 in (6.4.1) in the book.
   *
   * @param x_est_prev The previous estimated state of the GaussMix_x.
   * @param dt The time step.
   * @return The mixing probabilities. Each element is mixing_probs[s_{k-1}, s_k] = mu_{s_{k-1}|s_k}
   */
  Mat_nn calculate_mixing_probs(const GaussMix_x &x_est_prev, double dt)
  {
    Mat_nn pi_mat      = imm_model_.get_pi_mat_d(dt);
    Vec_n prev_weights = x_est_prev.weights();

    // mu_{s_{k-1}|s_k} = pi_{s_{k-1}|s_k} * mu_{s_{k-1}|k-1}
    Mat_nn mixing_probs = pi_mat.cwiseProduct(prev_weights.replicate(1, N_MODELS));

    // Normalize
    for (int i = 0; i < mixing_probs.cols(); i++) {
      mixing_probs.col(i) /= mixing_probs.col(i).sum();
    }
    return mixing_probs;
  }

  /**
   * @brief Calculate moment-based approximation, following step 2 in (6.4.1) in the book
   * @param x_est_prev Mixture from previous time step
   * @param mixing_probs Mixing probabilities
   * @return GaussMix_x Moment-based predictions
   */
  GaussMix_x mixing(const GaussMix_x x_est_prevs, Mat_nn mixing_probs)
  {
    GaussMix_x moment_based_preds;
    for (int i = 0; i < N_MODELS; i++) {
      GaussMix_x mixture(mixing_probs.row(i), x_est_prevs.gaussians());
      moment_based_preds += 1.0 / N_MODELS * mixture.reduce();
    }
    return moment_based_preds;
  }

  /**
   * @brief Calculate the filter outputs for each mode (6.36), following step 3 in (6.4.1) in the book
   * @param moment_based_preds Moment-based predictions
   * @param z_meas Vec_z Measurement
   * @param dt double Time step
   */
  std::array<std::tuple<Gauss_x, Gauss_x, Gauss_z>, N_MODELS> mode_matched_filter(const std::array<Gauss_x, N_MODELS> &moment_based_preds, const Vec_z &z_meas,
                                                                                  double dt)
  {
    std::array<std::tuple<Gauss_x, Gauss_x, Gauss_z>, N_MODELS> ekf_outs;

    // Can't loop through a tuple normally, so we use std::apply
    std::apply(
        [this, &ekf_outs, &moment_based_preds, &z_meas, &dt](auto &&...modelPtr) {
          size_t i = 0;
          (((ekf_outs[i] = filter::EKF_M<decltype(*modelPtr), SensModT>::step(modelPtr, sensor_model_, moment_based_preds[i], z_meas, dt)), i++), ...);
        },
        imm_model_.get_models());
    return ekf_outs;
  }

  /**
   * @brief Update the mixing probabilites using (6.37) from step 3 and (6.38) from step 4 in (6.4.1) in the book
   * @param z_preds Mode-match filter outputs
   * @param z_meas Vec_z Measurement
   * @param dt double Time step
   * @param weights Vec_n Weights
   */
  Mat_nn update_probabilities(const std::vector<Gauss_z> &z_preds, const Vec_z &z_meas, double dt, const Vec_n &weights)
  {
    Mat_nn pi_mat      = imm_model_.get_pi_mat_d(dt);
    Vec_n weights_pred = pi_mat.transpose() * weights;
    Vec_n z_probs      = Vec_n::Zero();

    for (int i = 0; i < N_MODELS; i++) {
      z_probs(i) = z_preds.at(i).pdf(z_meas);
    }

    Vec_n weights_upd = z_probs.cwiseProduct(weights_pred);
    weights_upd /= weights_upd.sum();

    return weights_upd;
  }

  /**
   * @brief Perform one IMM filter step
   * @param x_est_prev Mixture from previous time step
   * @param z_meas Vec_z
   * @param dt Time step
   */
  std::tuple<GaussMix_x, GaussMix_x, GaussMix_z> step(const GaussMix_x &x_est_prev, Vec_z z_meas, double dt)
  {
    Mat_nn mixing_probs                         = calculate_mixing_probs(x_est_prev, dt);
    GaussMix_x moment_based_preds               = mixing(x_est_prev, mixing_probs);
    auto [x_est_upds, x_est_preds, z_est_preds] = mode_matched_filter(moment_based_preds, z_meas, dt);
    Vec_n weights_upd                           = update_probabilities(z_est_preds, z_meas, dt, mixing_probs.row(0));

    GaussMix_x x_est_upd(weights_upd, x_est_upds);
    GaussMix_x x_est_pred(x_est_prev.weights, x_est_preds);
    GaussMix_z z_est_pred(x_est_prev.weights, z_est_preds);

    return {x_est_upd, x_est_pred, z_est_pred};
  }

private:
  ImmModelT imm_model_;
  SensModTPtr sensor_model_;
};

} // namespace filter
} // namespace vortex