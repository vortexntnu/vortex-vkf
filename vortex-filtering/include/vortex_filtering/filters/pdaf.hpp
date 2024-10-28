#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <string>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

namespace vortex::filter {

namespace config {
struct PDAF {
  double mahalanobis_threshold = 1.0;
  double min_gate_threshold    = 0.0;
  double max_gate_threshold    = std::numeric_limits<double>::max();
  double prob_of_detection     = 1.0;
  double clutter_intensity     = 1.0;
};
} // namespace config

template <concepts::DynamicModelLTVWithDefinedSizes DynModT, concepts::SensorModelLTVWithDefinedSizes SensModT> class PDAF {
public:
  static constexpr int N_DIM_x = DynModT::N_DIM_x;
  static constexpr int N_DIM_z = SensModT::N_DIM_z;
  static constexpr int N_DIM_u = DynModT::N_DIM_u;
  static constexpr int N_DIM_v = DynModT::N_DIM_v;
  static constexpr int N_DIM_w = SensModT::N_DIM_w;

  using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  using Gauss_z    = typename T::Gauss_z;
  using Gauss_x    = typename T::Gauss_x;
  using Vec_z      = typename T::Vec_z;
  using GaussMix_x = typename T::GaussMix_x;
  using GaussMix_z = typename T::GaussMix_z;
  using Arr_zXd    = Eigen::Array<double, N_DIM_z, Eigen::Dynamic>;
  using Arr_1Xb    = Eigen::Array<bool, 1, Eigen::Dynamic>;
  using Gauss_xX   = std::vector<Gauss_x>;
  using EKF        = vortex::filter::EKF_t<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  struct Config {
    config::PDAF pdaf;
  };

  struct Output {
    Gauss_x x_;
    Gauss_x x_prediction;
    Gauss_z z_prediction;
    Gauss_xX x_updates;
    Arr_1Xb gated_measurements;
  };

  PDAF() = delete;

  /**
   * @brief Perform one step of the Probabilistic Data Association Filter
   * 
   * @param dyn_model The dynamic model
   * @param sen_model The sensor model
   * @param timestep Time step in seconds
   * @param x_est The estimated state
   * @param z_measurements Array of measurements
   * @param config Configuration for the PDAF
   * @return `Output` The result of the PDAF step and some intermediate results
   */
  static Output step(const DynModT &dyn_model, const SensModT &sen_model, double timestep, const Gauss_x &x_est, const Arr_zXd &z_measurements,
                     const Config &config)
  {
    auto [x_pred, z_pred]   = EKF::predict(dyn_model, sen_model, timestep, x_est);
    auto gated_measurements = apply_gate(z_measurements, z_pred, config);
    auto inside_meas        = get_inside_measurements(z_measurements, gated_measurements);

    Gauss_xX x_updated;
    for (const auto &z_k : inside_meas.colwise()) {
      x_updated.push_back(EKF::update(sen_model, x_pred, z_pred, z_k));
    }

    Gauss_x x_final = get_weighted_average(inside_meas, x_updated, z_pred, x_pred, config.pdaf.prob_of_detection, config.pdaf.clutter_intensity);
    return {x_final, x_pred, z_pred, x_updated, gated_measurements};
  }

  /**
   * @brief Apply gate to the measurements
   * 
   * @param z_measurements Array of measurements
   * @param z_pred Predicted measurement
   * @param config Configuration for the PDAF
   * @return `Arr_1Xb` Indeces of the measurements that are inside the gate
   */
  static Arr_1Xb apply_gate(const Arr_zXd &z_measurements, const Gauss_z &z_pred, Config config)
  {
    double mahalanobis_threshold = config.pdaf.mahalanobis_threshold;
    double min_gate_threshold    = config.pdaf.min_gate_threshold;
    double max_gate_threshold    = config.pdaf.max_gate_threshold;

    Arr_1Xb gated_measurements(1, z_measurements.cols());

    for (size_t a_k = 0; const Vec_z &z_k : z_measurements.colwise()) {
      double mahalanobis_distance = z_pred.mahalanobis_distance(z_k);
      double regular_distance     = (z_pred.mean() - z_k).norm();
      gated_measurements(a_k++) =
          (mahalanobis_distance <= mahalanobis_threshold || regular_distance <= min_gate_threshold) && regular_distance <= max_gate_threshold;
    }
    return gated_measurements;
  }

  /**
   * @brief Get the measurements that are inside the gate
   * 
   * @param z_measurements Array of measurements
   * @param gated_measurements Indeces of the measurements that are inside the gate
   * @return `Arr_zXd` The measurements that are inside the gate
   */
  static Arr_zXd get_inside_measurements(const Arr_zXd &z_measurements, const Arr_1Xb &gated_measurements)
  {
    Arr_zXd inside_meas(N_DIM_z, gated_measurements.count());
    for (size_t i = 0, j = 0; bool gated : gated_measurements) {
      if (gated) {
        inside_meas.col(j++) = z_measurements.col(i);
      }
      i++;
    }
    return inside_meas;
  }

  /**
   * @brief Get the weighted average of the states
   * 
   * @param z_measurements Array of measurements
   * @param updated_states Array of updated states
   * @param z_pred Predicted measurement
   * @param x_pred Predicted state
   * @param prob_of_detection Probability of detection
   * @param clutter_intensity Clutter intensity
   * @return `Gauss_x` The weighted average of the states
   */
  static Gauss_x get_weighted_average(const Arr_zXd &z_measurements, const Gauss_xX &updated_states, const Gauss_z &z_pred, const Gauss_x &x_pred,
                                      double prob_of_detection, double clutter_intensity)
  {
    Gauss_xX states;
    states.push_back(x_pred);
    states.insert(states.end(), updated_states.begin(), updated_states.end());

    Eigen::VectorXd weights = get_weights(z_measurements, z_pred, prob_of_detection, clutter_intensity);

    return GaussMix_x{weights, states}.reduce();
  }

  /**
   * @brief Get the weights for the measurements
   * 
   * @param z_measurements Array of measurements
   * @param z_pred Predicted measurement
   * @param prob_of_detection Probability of detection
   * @param clutter_intensity Clutter intensity
   * @return `Eigen::VectorXd` The weights for the measurements
   */
  static Eigen::VectorXd get_weights(const Arr_zXd &z_measurements, const Gauss_z &z_pred, double prob_of_detection, double clutter_intensity)
  {
    double lambda = clutter_intensity;
    double P_d    = prob_of_detection;

    Eigen::VectorXd weights(z_measurements.cols() + 1);

    // in case no measurement assosiates with the target
    weights(0) = lambda * (1 - P_d);

    // measurements associating with the target
    for (size_t a_k = 1; const Vec_z &z_k : z_measurements.colwise()) {
      weights(a_k++) = P_d * z_pred.pdf(z_k);
    }

    // normalize weights
    weights /= weights.sum();

    return weights;
  }

  /**
   * @brief Get association probabilities according to Corollary 7.3.3
   * 
   * @param z_likelyhoods Array of likelyhoods
   * @param prob_of_detection Probability of detection
   * @param clutter_intensity Clutter intensity
   * @return `Eigen::VectorXd` The weights for the measurements
   */
  static Eigen::ArrayXd association_probabilities(const Eigen::ArrayXd &z_likelyhoods, double prob_of_detection, double clutter_intensity)
  {
    size_t m_k    = z_likelyhoods.size();
    double lambda = clutter_intensity;
    double P_d    = prob_of_detection;

    Eigen::ArrayXd weights(m_k + 1);

    // Accociation probabilities (Corrolary 7.3.3)
    weights(0)        = lambda * (1 - P_d);
    weights.tail(m_k) = P_d * z_likelyhoods;

    // normalize weights
    weights /= weights.sum();

    return weights;
  }
};

} // namespace vortex::filter