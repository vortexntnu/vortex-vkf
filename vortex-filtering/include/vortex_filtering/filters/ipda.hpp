#pragma once
#include <Eigen/Dense>
#include <vector>
#include <vortex_filtering/filters/pdaf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

namespace vortex::filter {

namespace config {
struct IPDA {
  double prob_of_survival                           = 0.99;
  bool estimate_clutter                             = true;
  bool update_existence_probability_on_no_detection = true;
};
} // namespace config

template <concepts::DynamicModelLTVWithDefinedSizes DynModT, concepts::SensorModelLTVWithDefinedSizes SensModT> class IPDA {
public:
  static constexpr int N_DIM_x = DynModT::N_DIM_x;
  static constexpr int N_DIM_z = SensModT::N_DIM_z;
  static constexpr int N_DIM_u = DynModT::N_DIM_u;
  static constexpr int N_DIM_v = DynModT::N_DIM_v;
  static constexpr int N_DIM_w = SensModT::N_DIM_w;

  using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

  using Arr_zXd = Eigen::Array<double, N_DIM_z, Eigen::Dynamic>;
  using Arr_1Xb = Eigen::Array<bool, 1, Eigen::Dynamic>;
  using EKF     = vortex::filter::EKF_t<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;
  using PDAF    = vortex::filter::PDAF<DynModT, SensModT>;

  IPDA() = delete;

  struct Config {
    config::PDAF pdaf;
    config::IPDA ipda;
  };

  struct State {
    T::Gauss_x x_estimate;
    double existence_probability;
  };

  struct Output {
    State state;
    T::Gauss_x x_prediction;
    T::Gauss_z z_prediction;
    std::vector<typename T::Gauss_x> x_updates;
    Arr_1Xb gated_measurements;
  };

  static double existence_prediction(double existence_prob_est, double prob_of_survival)
  {
    double r_km1 = existence_prob_est;
    double P_s   = prob_of_survival;
    return P_s * r_km1; // (7.28)
  }

  /**
   * @brief Calculates the existence probability given the measurements and the previous existence probability.
   * @param z_measurements The measurements to iterate over.
   * @param z_pred The predicted measurement.
   * @param existence_prob_est (r_{k-1}) The previous existence probability.
   * @param config The configuration for the IPDA.
   * @return The existence probability.
   */
  static double existence_prob_update(const Arr_zXd &z_measurements, T::Gauss_z &z_pred, double existence_prob_pred, Config config)
  {
    double r_kgkm1 = existence_prob_pred;
    double P_d     = config.pdaf.prob_of_detection;
    double lambda  = config.pdaf.clutter_intensity;

    // predicted measurement probability
    double z_pred_prob = 0.0;
    for (const typename T::Vec_z &z_k : z_measurements.colwise()) {
      z_pred_prob += z_pred.pdf(z_k);
    }

    // posterior existence probability r_k
    double L_k = 1 - P_d + P_d / lambda * z_pred_prob;        // (7.33)
    double r_k = (L_k * r_kgkm1) / (1 - (1 - L_k) * r_kgkm1); // (7.32)
    return r_k;
  }

  /**
   * @brief Calculates the existence probability given the likelyhood of the measurements and the previous existence probability.
   * @param z_likelyhoods (l_a_k) The likelyhood of the measurements
   * @param existence_prob_est (r_{k-1}) The previous existence probability.
   * @param config The configuration for the IPDA.
   * @return The existence probability (r_k).
   */
  static double existence_prob_update(const Eigen::ArrayXd z_likelyhoods, double existence_prob_pred, Config config)
  {
    double r_kgkm1 = existence_prob_pred; // r_k given k minus 1
    double P_d     = config.pdaf.prob_of_detection;
    double lambda  = config.pdaf.clutter_intensity;

    // posterior existence probability r_k
    double L_k = 1 - P_d + P_d / lambda * z_likelyhoods.sum(); // (7.33)
    double r_k = (L_k * r_kgkm1) / (1 - (1 - L_k) * r_kgkm1);  // (7.32)
    return r_k;
  }

  /**
   * @brief Estimates the clutter intensity using (7.31)
   * @param z_pred The predicted measurement.
   * @param predicted_existence_probability (r_{k|k-1})  The predicted existence probability.
   * @param num_measurements (m_k) The number of z_measurements.
   * @param config The configuration for the IPDA.
   * @return The clutter intensity.
   */
  static double estimate_clutter_intensity(const T::Gauss_z &z_pred, double predicted_existence_probability, double num_measurements, Config config)
  {
    size_t m_k = num_measurements;
    double P_d = config.pdaf.prob_of_detection;
    double r_k = predicted_existence_probability;
    // TODO: make this work for N_DIM_z /= 2
    static_assert(N_DIM_z == 2);
    double V_k = utils::Ellipse(z_pred, config.pdaf.mahalanobis_threshold).area(); // gate area

    if (m_k == 0) {
      return 0.0;
    }
    return 1 / V_k * (m_k - r_k * P_d); // (7.31)
  }

  static Output step(const DynModT &dyn_mod, const SensModT &sens_mod, double timestep, const State &state_est_prev, const Arr_zXd &z_measurements,
                     Config &config)
  {
    double existence_prob_pred = existence_prediction(state_est_prev.existence_probability, config.ipda.prob_of_survival);

    if (config.ipda.estimate_clutter) {
      typename T::Gauss_z z_pred;
      std::tie(std::ignore, z_pred) = EKF::predict(dyn_mod, sens_mod, timestep, state_est_prev.x_estimate);
      config.pdaf.clutter_intensity = estimate_clutter_intensity(z_pred, existence_prob_pred, z_measurements.cols(), config);
    }

    auto [x_post, x_pred, z_pred, x_upd, gated_measurements] =
        PDAF::step(dyn_mod, sens_mod, timestep, state_est_prev.x_estimate, z_measurements, {config.pdaf});

    Arr_zXd z_meas_inside = PDAF::get_inside_measurements(z_measurements, gated_measurements);

    double existence_probability_upd = existence_prob_pred;
    if (z_measurements.cols() == 0 && !config.ipda.update_existence_probability_on_no_detection) {
      existence_probability_upd = existence_prob_update(z_meas_inside, z_pred, existence_prob_pred, config);
    }
    // clang-format off
    return {
      .state = {
        .x_estimate            = x_post,
        .existence_probability = existence_probability_upd,
      },
      .x_prediction       = x_pred,
      .z_prediction       = z_pred,
      .x_updates          = x_upd,
      .gated_measurements = gated_measurements
    };
    // clang-format on
  }
};
} // namespace vortex::filter