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

  using Gauss_z    = typename T::Gauss_z;
  using Gauss_x    = typename T::Gauss_x;
  using Vec_z      = typename T::Vec_z;
  using GaussMix_x = typename T::GaussMix_x;
  using Mat_zm_k   = Eigen::Matrix<double, N_DIM_z, Eigen::Dynamic>;
  using PDAF       = vortex::filter::PDAF<vortex::models::ConstantVelocity, vortex::models::IdentitySensorModel<4, 2>>;

  IPDA() = delete;

  struct Config : public PDAF::Config
  {
    double prob_of_survival                           = 0.99;
    bool estimate_clutter                             = true;
    bool update_existence_probability_on_no_detection = true;
  };

  static double existence_prediction(double existence_prob_est, double prob_of_survival)
  {
    double r_km1 = existence_prob_est;
    double P_s   = prob_of_survival;
    return P_s * r_km1; // (7.28)
  }

  /**
   * @brief Calculates the existence probability of an object.
   * @param z_measurements The measurements to iterate over.
   * @param z_pred The predicted measurement.
   * @param existence_prob_est (r_{k-1}) The previous existence probability.
   * @param config The configuration for the IPDA.
   * @return The existence probability.
   */
  static double existence_prob_update(const Mat_zm_k &z_measurements, Gauss_z &z_pred, double existence_prob_pred, Config config)
  {
    double r_kgkm1 = existence_prob_pred;
    double P_s     = config.prob_of_survival;
    double P_d     = config.prob_of_detection;
    double m_k     = z_measurements.size();

    // predicted measurement probability
    double z_pred_prob = 0.0;
    for (const Vec_z &z_k : z_measurements.colwise()) {
      z_pred_prob += z_pred.pdf(z_k);
    }

    // posterior existence probability r_k
    double L_k = 1 - P_d + P_d / lambda * z_pred_prob;        // (7.33)
    double r_k = (L_k * r_kgkm1) / (1 - (1 - L_k) * r_kgkm1); // (7.32)
    return r_k;
  }

  /**
   * @brief Calculates the existence probability of an object.
   * @param z_likelyhoods (l_a_k) The likelyhood of the measurements
   * @param existence_prob_est (r_{k-1}) The previous existence probability.
   * @param config The configuration for the IPDA.
   * @return The existence probability (r_k).
   */
  static double existence_prob_update(const Eigen::VectorXd z_likelyhoods, double existence_prob_pred, Config config)
  {
    double r_kgkm1 = existence_prob_pred; // r_k given k minus 1
    double P_s     = config.prob_of_survival;
    double P_d     = config.prob_of_detection;
    double m_k     = z_measurements.size();

    // posterior existence probability r_k
    double L_k = 1 - P_d + P_d / lambda * z_likelyhoods.sum(); // (7.33)
    double r_k = (L_k * r_kgkm1) / (1 - (1 - L_k) * r_kgkm1);  // (7.32)
    return r_k;
  }

  /**
   * @brief Estimates the clutter intensity using (7.31)
   * @param z_pred The predicted measurement.
   * @param predicted_existence_probability (r_{k|k-1})  The predicted existence probability.
   * @param prob_of_detection (P_d) The probability of detection.
   * @param num_measurements (m_k) The number of z_measurements.
   * @return The clutter intensity.
   */
  static double estimate_clutter_intensity(const Gauss_z &z_pred, double predicted_existence_probability, double probability_of_detection,
                                           double num_measurements)
  {
    size_t m_k = num_measurements;
    double P_d = probability_of_detection;
    double r_k = predicted_existence_probability;
    // TODO: make this work for N_DIM_z /= 2
    static_assert(N_DIM_z == 2) double V_k = utils::Ellipse(z_pred, config.mahalanobis_threshold).area(); // gate area

    if (m_k == 0) {
      return 0.0;
    }
    return 1 / V_k * (m_k - r_k * P_d); // (7.31)
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
  step(const DynModT &dyn_mod, const SensModT &sens_mod, double timestep, const Gauss_x &x_est, const std::vector<Vec_z> &z_meas, double existence_prob_est,
       IPDA::Config &config)
  {
    double existence_prop_pred = existence_prediction(existence_prob_est, config.prob_of_survival);

    if (config.estimate_clutter) {
      config.clutter_intensity = estimate_clutter_intensity(z_pred, existence_prob_pred, config.prob_of_detection, z_meas.size());
    }

    auto [x_post, z_meas_inside, z_meas_outside, x_pred, z_pred, x_upd] =
        PDAF::step(dyn_mod, sens_mod, timestep, x_est, z_meas, static_cast<PDAF::Config>(config));

    Eigen::Matrix2i z_measurements(z_meas.size(), 2);
    double existence_probability_upd = existence_prop_pred;
    if (z_measurements.cols() == 0 && !config.update_existence_probability_on_no_detection) {
      existence_probability_upd = existence_prob_update(z_meas_inside, z_pred, existence_prob_est, config);
    }
    return {x_post, existence_probability_upd, z_meas_inside, z_meas_outside, x_pred, z_pred, x_upd};
  }
};
}  // namespace vortex::filter
