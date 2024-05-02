#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

namespace vortex::filter {

template <concepts::DynamicModelLTVWithDefinedSizes DynModT, concepts::SensorModelLTVWithDefinedSizes SensModT> class PDAF {
public:
  static constexpr int N_DIM_x = DynModT::N_DIM_x;
  static constexpr int N_DIM_z = SensModT::N_DIM_z;
  static constexpr int N_DIM_u = DynModT::N_DIM_u;
  static constexpr int N_DIM_v = DynModT::N_DIM_v;
  static constexpr int N_DIM_w = SensModT::N_DIM_w;

  using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  using EKF            = vortex::filter::EKF_t<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;
  using Gauss_z        = typename T::Gauss_z;
  using Gauss_x        = typename T::Gauss_x;
  using Vec_z          = typename T::Vec_z;
  using Arr_zm_k       = Eigen::Array<double, N_DIM_z, Eigen::Dynamic>;
  using StatesXd       = std::vector<Gauss_x>;
  using GaussMixZd     = vortex::prob::GaussianMixture<N_DIM_x>;

  struct Config {
    double mahalanobis_threshold = 1.0;
    double min_gate_threshold    = 0.0;
    double max_gate_threshold    = HUGE_VAL;
    double prob_of_detection     = 1.0;
    double clutter_intensity     = 1.0;
  };

  PDAF() = delete;

  static std::tuple<Gauss_x, Arr_zm_k, Arr_zm_k, Gauss_x, Gauss_z, StatesXd> step(const DynModT &dyn_model, const SensModT &sen_model, double timestep,
                                                                                  const Gauss_x &x_est, const Arr_zm_k &z_measurements, const Config &config)
  {
    auto [x_pred, z_pred]  = EKF::predict(dyn_model, sen_model, timestep, x_est);
    auto [inside_meas, outside_meas] = apply_gate(z_measurements, z_pred, config.mahalanobis_threshold, config.min_gate_threshold, config.max_gate_threshold);

    StatesXd x_updated;
    for (const auto &z_k : inside_meas.colwise()) {
      x_updated.push_back(EKF::update(sen_model, x_pred, z_pred, z_k));
    }

    Gauss_x x_final = get_weighted_average(inside_meas, x_updated, z_pred, x_pred, config.prob_of_detection, config.clutter_intensity);
    return {x_final, inside_meas, outside_meas, x_pred, z_pred, x_updated};
  }

  static std::tuple<Arr_zm_k, Arr_zm_k> apply_gate(const Arr_zm_k &z_measurements, const Gauss_z &z_pred, double mahalanobis_threshold,
                                                   double min_gate_threshold = 0.0, double max_gate_threshold = HUGE_VAL)
  {
    Arr_zm_k inside_meas(SensModT::N_DIM_z, 0);
    Arr_zm_k outside_meas(SensModT::N_DIM_z, 0);

    for (const Vec_z &z_k : z_measurements.colwise()) {
      double mahalanobis_distance = z_pred.mahalanobis_distance(z_k);
      double regular_distance     = (z_pred.mean() - z_k).norm();
      if ((mahalanobis_distance <= mahalanobis_threshold || regular_distance <= min_gate_threshold) && regular_distance <= max_gate_threshold) {
        inside_meas.conservativeResize(Eigen::NoChange, inside_meas.cols() + 1);
        inside_meas.rightCols(1) = z_k;
      }
      else {
        outside_meas.conservativeResize(Eigen::NoChange, outside_meas.cols() + 1);
        outside_meas.rightCols(1) = z_k;
      }
    }

    return {inside_meas, outside_meas};
  }

  // Getting weighted average of the predicted states
  static Gauss_x get_weighted_average(const Arr_zm_k &z_measurements, const StatesXd &updated_states, const Gauss_z &z_pred, const Gauss_x &x_pred,
                                      double prob_of_detection, double clutter_intensity)
  {
    StatesXd states;
    states.push_back(x_pred);
    states.insert(states.end(), updated_states.begin(), updated_states.end());

    Eigen::VectorXd weights = get_weights(z_measurements, z_pred, prob_of_detection, clutter_intensity);

    GaussMixZd gaussian_mixture(weights, states);

    return gaussian_mixture.reduce();
  }

  // Getting association probabilities according to textbook p. 123 "Corollary 7.3.3"
  static Eigen::VectorXd get_weights(const Arr_zm_k &z_measurements, const Gauss_z &z_pred, double prob_of_detection, double clutter_intensity)
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

  static Eigen::ArrayXd association_probabilities(const Eigen::ArrayXd &z_likelyhoods, double prob_of_detection, double clutter_intensity)
  {
    size_t m_k    = z_likelyhoods.size();
    double lambda = clutter_intensity;
    double P_d    = prob_of_detection;

    Eigen::ArrayXd weights(m_k + 1);

    // Accociation probabilities (Corrolary 7.3.3)
    weights(0) = lambda * (1 - P_d);         
    weights.tail(m_k) = P_d * z_likelyhoods;

    // normalize weights
    weights /= weights.sum();

    return weights;

  }
};

} // namespace vortex::filter