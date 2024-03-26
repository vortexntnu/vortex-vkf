#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
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
  using MeasurementsZd = std::vector<Vec_z>;
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

  static std::tuple<Gauss_x, MeasurementsZd, MeasurementsZd, Gauss_x, Gauss_z, StatesXd>
  step(const DynModT &dyn_model, const SensModT &sen_model, double timestep, const Gauss_x &x_est, const MeasurementsZd &z_meas, const Config &config)
  {
    auto [x_pred, z_pred]  = EKF::predict(dyn_model, sen_model, timestep, x_est);
    auto [inside, outside] = apply_gate(z_meas, z_pred, config.mahalanobis_threshold, config.min_gate_threshold, config.max_gate_threshold);

    StatesXd x_updated;
    for (const auto &measurement : inside) {
      x_updated.push_back(EKF::update(sen_model, x_pred, z_pred, measurement));
    }

    Gauss_x x_final = get_weighted_average(inside, x_updated, z_pred, x_pred, config.prob_of_detection, config.clutter_intensity);
    return {x_final, inside, outside, x_pred, z_pred, x_updated};
  }

  static std::tuple<MeasurementsZd, MeasurementsZd> apply_gate(const MeasurementsZd &z_meas, const Gauss_z &z_pred, double mahalanobis_threshold,
                                                               double min_gate_threshold = 0.0, double max_gate_threshold = HUGE_VAL)
  {
    MeasurementsZd inside_meas;
    MeasurementsZd outside_meas;

    for (const auto &measurement : z_meas) {
      double mahalanobis_distance = z_pred.mahalanobis_distance(measurement);
      double regular_distance     = (z_pred.mean() - measurement).norm();
      if ((mahalanobis_distance <= mahalanobis_threshold || regular_distance <= min_gate_threshold) && regular_distance <= max_gate_threshold) {
        inside_meas.push_back(measurement);
      }
      else {
        outside_meas.push_back(measurement);
      }
    }

    return {inside_meas, outside_meas};
  }

  // Getting weighted average of the predicted states
  static Gauss_x get_weighted_average(const MeasurementsZd &z_meas, const StatesXd &updated_states, const Gauss_z &z_pred, const Gauss_x &x_pred,
                                      double prob_of_detection, double clutter_intensity)
  {
    StatesXd states;
    states.push_back(x_pred);
    states.insert(states.end(), updated_states.begin(), updated_states.end());

    Eigen::VectorXd weights = get_weights(z_meas, z_pred, prob_of_detection, clutter_intensity);

    GaussMixZd gaussian_mixture(weights, states);

    return gaussian_mixture.reduce();
  }

  // Getting association probabilities according to textbook p. 123 "Corollary 7.3.3"
  static Eigen::VectorXd get_weights(const MeasurementsZd &z_meas, const Gauss_z &z_pred, double prob_of_detection, double clutter_intensity)
  {
    Eigen::VectorXd weights(z_meas.size() + 1);

    // in case no measurement assosiates with the target
    double no_association = clutter_intensity * (1 - prob_of_detection);
    weights(0)            = no_association;

    // measurements associating with the target
    for (size_t k = 1; k < z_meas.size() + 1; k++) {
      weights(k) = (prob_of_detection * z_pred.pdf(z_meas.at(k - 1)));
    }

    // normalize weights
    weights /= weights.sum();

    return weights;
  }
};

} // namespace vortex::filter