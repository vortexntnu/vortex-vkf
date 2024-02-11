#pragma once
#include <vector>
#include <Eigen/Dense>
#include <vortex_filtering/filters/pdaf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

namespace vortex::filter
{
template <models::concepts::DynamicModelLTV DynModT, models::concepts::SensorModelLTV SensModT>
class IPDA
{
public:
  using SensModI = typename SensModT::SensModI;
  using DynModI = typename DynModT::DynModI;
  using DynModPtr = std::shared_ptr<DynModI>;
  using SensModPtr = std::shared_ptr<SensModI>;
  using EKF = vortex::filter::EKF<DynModI, SensModI>;
  using Gauss_z = typename SensModI::Gauss_z;
  using Gauss_x = typename DynModI::Gauss_x;
  using Vec_z = typename SensModI::Vec_z;
  using GaussMix = vortex::prob::GaussianMixture<DynModI::N_DIM_x>;
  using PDAF = vortex::filter::PDAF<vortex::models::ConstantVelocity<2>, vortex::models::IdentitySensorModel<4, 2>>;
  IPDA() = delete;
  /// @brief
  /// @param measurements Measurements to iterate over
  /// @param probability_of_survival How likely the object is to survive (Ps)
  /// @param probability_of_detection How likely the object is to be detected (Pd)
  /// @param clutter_intensity How likely it is to have a false positive
  /// @param z_pred The predicted measurement
  /// @return The existence probability
  static double get_existence_probability(const std::vector<Vec_z>& measurements, double probability_of_survival,
                                          double last_detection_probability_, double probability_of_detection,
                                          double clutter_intensity, Gauss_z& z_pred)
  {
    double predicted_existence_probability = probability_of_survival * last_detection_probability_;  // Finds Ps

    double summed = 0;
    for (const Vec_z& measurement : measurements)
    {
      summed += z_pred.pdf(measurement);
    }
    double l_k = 1 - probability_of_detection + probability_of_detection / clutter_intensity * summed;

    return (l_k * predicted_existence_probability) / (1 - (1 - l_k) * predicted_existence_probability);
  }

  static std::tuple<Gauss_x, double, std::vector<Vec_z>, std::vector<Vec_z>, Gauss_x, Gauss_z, std::vector<Gauss_x>>
  step(const Gauss_x& x_est, const std::vector<Vec_z>& z_meas, double timestep, const DynModPtr& dyn_model,
       const SensModPtr& sen_model, double gate_threshold, double prob_of_detection, double prob_of_survival,
       double survive_est, double clutter_intensity)
  {
    auto [x_pred, z_pred] = EKF::predict(dyn_model, sen_model, timestep, x_est);
    auto [inside, outside] = PDAF::apply_gate(z_meas, z_pred, gate_threshold);

    std::vector<Gauss_x> x_updated;
    for (const auto& measurement : inside)
    {
      x_updated.push_back(EKF::update(sen_model, x_pred, z_pred, measurement));
    }

    Gauss_x x_final =
        PDAF::get_weighted_average(inside, x_updated, z_pred, x_pred, prob_of_detection, clutter_intensity);

    double existence_probability =
        get_existence_probability(inside, prob_of_survival, survive_est, prob_of_detection, clutter_intensity, z_pred);
    return { x_final, existence_probability, inside, outside, x_pred, z_pred, x_updated };
  }
};
}  // namespace vortex::filter
