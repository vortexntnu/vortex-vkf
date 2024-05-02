/**
 * @file ipda.hpp
 * @author Tristan Wolfram
 * @brief File for the PDAF filter
 * @version 1.0
 * @date 2024-05-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

namespace vortex::filter
{
/**
 * @brief The PDAF filter class
 * @tparam DynModT The dynamic model
 * @tparam SensModT The sensor model
 */
template <concepts::DynamicModelLTVWithDefinedSizes DynModT, concepts::SensorModelLTVWithDefinedSizes SensModT>
class PDAF
{
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

  struct Config
  {
    double mahalanobis_threshold = 1.0;
    double min_gate_threshold = 0.0;
    double max_gate_threshold = HUGE_VAL;
    double prob_of_detection = 1.0;
    double clutter_intensity = 1.0;
  };

  PDAF() = delete;

  /**
   * @brief The step function for the PDAF filter
   * @param dyn_model The dynamic model
   * @param sen_model The sensor model
   * @param timestep The timestep
   * @param x_est The estimated state (current state)
   * @param z_meas The measurements
   * @param config The configuration
   * @return A tuple containing the (new) estimated state, the measurements that were inside the gate, the measurements
   * that were outside the gate, the predicted state (from dynamic model), the predicted measurement (from sensor
   * model), and the updated states (comparisson of prediced state and actual measurements)
   */
  static std::tuple<Gauss_x, MeasurementsZd, MeasurementsZd, Gauss_x, Gauss_z, StatesXd>
  step(const DynModT& dyn_model, const SensModT& sen_model, double timestep, const Gauss_x& x_est,
       const MeasurementsZd& z_meas, const Config& config)
  {
    auto [x_pred, z_pred] = EKF::predict(dyn_model, sen_model, timestep, x_est);
    auto [inside, outside] =
        apply_gate(z_meas, z_pred, config.mahalanobis_threshold, config.min_gate_threshold, config.max_gate_threshold);

    StatesXd x_updated;
    for (const auto& measurement : inside)
    {
      x_updated.push_back(EKF::update(sen_model, x_pred, z_pred, measurement));
    }

    Gauss_x x_final =
        get_weighted_average(inside, x_updated, z_pred, x_pred, config.prob_of_detection, config.clutter_intensity);
    return { x_final, inside, outside, x_pred, z_pred, x_updated };
  }

  /**
   * @brief Applies the gate to the measurements
   * @param z_meas The measurements
   * @param z_pred The predicted measurement
   * @param mahalanobis_threshold The threshold for the Mahalanobis distance
   * @param min_gate_threshold The minimum threshold for the gate
   * @param max_gate_threshold The maximum threshold for the gate
   * @return A tuple containing the measurements that were inside the gate and the measurements that were outside the
   * gate
   */
  static std::tuple<MeasurementsZd, MeasurementsZd> apply_gate(const MeasurementsZd& z_meas, const Gauss_z& z_pred,
                                                               double mahalanobis_threshold,
                                                               double min_gate_threshold = 0.0,
                                                               double max_gate_threshold = HUGE_VAL)
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

    return { inside_meas, outside_meas };
  }

  /**
   * @brief Calculates the weighted average of the measurements
   * @param z_meas The measurements
   * @param updated_states The updated states
   * @param z_pred The predicted measurement
   * @param x_pred The predicted state
   * @param prob_of_detection The probability of detection
   * @param clutter_intensity The clutter intensity
   * @return The weighted average of the measurements
   */
  static Gauss_x get_weighted_average(const MeasurementsZd& z_meas, const StatesXd& updated_states,
                                      const Gauss_z& z_pred, const Gauss_x& x_pred, double prob_of_detection,
                                      double clutter_intensity)
  {
    StatesXd states;
    states.push_back(x_pred);
    states.insert(states.end(), updated_states.begin(), updated_states.end());

    Eigen::VectorXd weights = get_weights(z_measurements, z_pred, prob_of_detection, clutter_intensity);

    GaussMixXd gaussian_mixture(weights, states);

    return gaussian_mixture.reduce();
  }

  /**
   * @brief Gets the weights for the measurements -> see association probabilities in textbook p. 123
   * "Corollary 7.3.3
   * @param z_meas The measurements
   * @param z_pred The predicted measurement
   * @param prob_of_detection The probability of detection
   * @param clutter_intensity The clutter intensity
   * @return The weights for the measurements -> same order as in z_meas
   * The first element is the weight for no association
   * length of weights = z_meas.size() + 1
   */
  static Eigen::VectorXd get_weights(const MeasurementsZd& z_meas, const Gauss_z& z_pred, double prob_of_detection,
                                     double clutter_intensity)
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

}  // namespace vortex::filter