/**
 * @file ipda.hpp
 * @author Tristan Wolfram
 * @brief File for the IPDA filter
 * @version 1.0
 * @date 2024-05-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#include <vector>
#include <Eigen/Dense>
#include <vortex_filtering/filters/pdaf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

namespace vortex::filter
{
/**
 * @brief The IPDA filter class
 * @tparam DynModT The dynamic model type
 * @tparam SensModT The sensor model type
 */
template <concepts::DynamicModelLTVWithDefinedSizes DynModT, concepts::SensorModelLTVWithDefinedSizes SensModT>
class IPDA
{
public:
  static constexpr int N_DIM_x = DynModT::N_DIM_x;
  static constexpr int N_DIM_z = SensModT::N_DIM_z;
  static constexpr int N_DIM_u = DynModT::N_DIM_u;
  static constexpr int N_DIM_v = DynModT::N_DIM_v;
  static constexpr int N_DIM_w = SensModT::N_DIM_w;

  using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  using EKF = vortex::filter::EKF_t<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  using Gauss_z = typename T::Gauss_z;
  using Gauss_x = typename T::Gauss_x;
  using Vec_z = typename T::Vec_z;
  using GaussMix = vortex::prob::GaussianMixture<N_DIM_x>;
  using PDAF = vortex::filter::PDAF<DynModT, SensModT>;

  IPDA() = delete;

  struct Config : public PDAF::Config
  {
    double prob_of_survival = 1.0;
  };

  /**
   * @brief Calculates the existence probability of an object.
   * @param measurements The measurements to iterate over.
   * @param probability_of_survival How likely the object is to survive (Ps).
   * @param last_detection_probability_ The last detection probability.
   * @param probability_of_detection How likely the object is to be detected (Pd).
   * @param clutter_intensity How likely it is to have a false positive.
   * @param z_pred The predicted measurement.
   * @return The existence probability.
   */
  static double get_existence_probability(const std::vector<Vec_z>& measurements, double probability_of_survival,
                                          double last_detection_probability_, double probability_of_detection,
                                          double clutter_intensity, Gauss_z& z_pred)
  {
    double predicted_existence_probability = probability_of_survival * last_detection_probability_;  // Finds Ps

    double sum = 0.0;
    for (const Vec_z& measurement : measurements)
    {
      sum += z_pred.pdf(measurement);
    }
    double l_k = 1 - probability_of_detection + probability_of_detection / clutter_intensity * sum;

    return (l_k * predicted_existence_probability) / (1 - (1 - l_k) * predicted_existence_probability);
  }

  /**
   * @brief The IPDAF step function. Gets following parameters and calculates the next state with the probablity of
   * existence.
   * @param dyn_model The dynamic model.
   * @param sen_model The sensor model.
   * @param timestep The timestep.
   * @param x_est The estimated state.
   * @param z_meas The percieved measurements.
   * @param existence_prob The estimated survival probability (current state).
   * @param config configuration data - see Config struct of PDAF.
   * @return A tuple containing the final state, the existence probability, the inside (of the gate) measurements, the
   * outside (of the gate) measurements, the predicted state, the predicted measurements, and the updated state.
   */
  static std::tuple<Gauss_x, double, std::vector<Vec_z>, std::vector<Vec_z>, Gauss_x, Gauss_z, std::vector<Gauss_x>>
  step(const DynModT& dyn_model, const SensModT& sen_model, double timestep, const Gauss_x& x_est,
       const std::vector<Vec_z>& z_meas, double existence_prob, const IPDA::Config& config)
  {
    auto [x_final, inside, outside, x_pred, z_pred, x_updated] =
        PDAF::step(dyn_model, sen_model, timestep, x_est, z_meas, static_cast<PDAF::Config>(config));

    double existence_probability = get_existence_probability(
        inside, config.prob_of_survival, existence_prob, config.prob_of_detection, config.clutter_intensity, z_pred);
    return { x_final, existence_probability, inside, outside, x_pred, z_pred, x_updated };
  }
};
}  // namespace vortex::filter
