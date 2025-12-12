#pragma once
#include <Eigen/Dense>
#include <vector>
#include <vortex_filtering/filters/pdaf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/utils/ellipsoid.hpp>

namespace vortex::filter {

namespace config {
struct IPDA {
    double prob_of_survival = 0.99;
    bool estimate_clutter = true;
    bool update_existence_probability_on_no_detection = true;
};
}  // namespace config

template <concepts::DynamicModelLTVWithDefinedSizes DynModT,
          concepts::SensorModelLTVWithDefinedSizes SensModT>
class IPDA {
   public:
    static constexpr int N_DIM_x = DynModT::N_DIM_x;
    static constexpr int N_DIM_z = SensModT::N_DIM_z;
    static constexpr int N_DIM_u = DynModT::N_DIM_u;
    static constexpr int N_DIM_v = DynModT::N_DIM_v;
    static constexpr int N_DIM_w = SensModT::N_DIM_w;

    using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

    using Gauss_z = typename T::Gauss_z;
    using Gauss_x = typename T::Gauss_x;
    using Gauss_xX = std::vector<Gauss_x>;

    using Arr_zXd = Eigen::Array<double, N_DIM_z, Eigen::Dynamic>;
    using Arr_1Xb = Eigen::Array<bool, 1, Eigen::Dynamic>;
    using EKF =
        vortex::filter::EKF_t<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;
    using PDAF = vortex::filter::PDAF<DynModT, SensModT>;

    IPDA() = delete;

    struct Config {
        config::PDAF pdaf;
        config::IPDA ipda;
    };

    struct State {
        Gauss_x x_estimate;
        double existence_probability;
    };

    struct Output {
        State state;
        Gauss_x x_prediction;
        Gauss_z z_prediction;
        Gauss_xX x_updates;
        Arr_1Xb gated_measurements;
    };

    /**
     * @brief Perform one IPDA prediction step
     *
     * @param dyn_mod The dynamic model
     * @param sens_mod The sensor model
     * @param dt Time step in seconds
     * @param state_est_prev The previous estimated state
     * @param z_measurements Array of measurements
     * @param config Configuration for the IPDA
     * @return `std::tuple<Gauss_x, Gauss_z, double>` The predicted state,
     * predicted measurement, and predicted existence probability
     */
    static std::tuple<Gauss_x, Gauss_z, double> predict(
        const DynModT dyn_mod,
        const SensModT& sens_mod,
        double dt,
        const State& state_est_prev,
        const Arr_zXd& z_measurements,
        Config& config) {
        double existence_prob_pred = existence_prediction(
            state_est_prev.existence_probability, config.ipda.prob_of_survival);

        auto [x_pred, z_pred] =
            EKF::predict(dyn_mod, sens_mod, dt, state_est_prev.x_estimate);

        if (config.ipda.estimate_clutter) {
            config.pdaf.clutter_intensity = estimate_clutter_intensity(
                z_pred, existence_prob_pred, z_measurements.cols(), config);
        }

        return {x_pred, z_pred, existence_prob_pred};
    }

    /**
     * @brief Performs one IPDA update step
     * @param sens_mod The sensor model
     * @param x_pred The predicted state
     * @param z_pred The predicted measurement
     * @param z_measurements Array of measurements
     * @param existence_prob_pred The predicted existence probability
     * @param config Configuration for the IPDA
     * @return `std::tuple<Gauss_x, Gauss_xX, Arr_1Xb, Arr_zXd, double>` The
     * updated state, all updated states for each measurement, the indices of
     * the measurements that are inside the gate, the measurements that are
     * inside the gate and the updated existence probability
     */
    static std::tuple<Gauss_x, Gauss_xX, Arr_1Xb, Arr_zXd, double> update(
        const SensModT& sens_mod,
        const Gauss_x& x_pred,
        const Gauss_z& z_pred,
        const Arr_zXd& z_measurements,
        double existence_prob_pred,
        const Config& config) {
        typename PDAF::Config pdaf_cfg{.pdaf = config.pdaf};
        auto [x_post, x_updates, gated_measurements, z_inside_meas,
              z_likelihoods] =
            PDAF::update(sens_mod, x_pred, z_pred, z_measurements, pdaf_cfg);

        double existence_prob_upd = existence_prob_pred;
        if (z_measurements.cols() > 0 ||
            config.ipda.update_existence_probability_on_no_detection) {
            existence_prob_upd = existence_prob_update(
                z_likelihoods, existence_prob_pred, config);
        }

        return std::make_tuple(x_post, x_updates, gated_measurements,
                               z_inside_meas, existence_prob_upd);
    }

    /**
     * @brief Perform one step of the Integrated Probabilistic Data Association
     * Filter
     *
     * @param dyn_mod The dynamic model
     * @param sens_mod The sensor model
     * @param dt Time step in seconds
     * @param state_est_prev The previous estimated state
     * @param z_measurements Array of measurements
     * @param config Configuration for the IPDA
     * @return `Output` The result of the IPDA step and some intermediate
     * results
     */
    static Output step(const DynModT& dyn_mod,
                       const SensModT& sens_mod,
                       double dt,
                       const State& state_est_prev,
                       const Arr_zXd& z_measurements,
                       Config& config) {
        auto [x_pred, z_pred, existence_prob_pred] = predict(
            dyn_mod, sens_mod, dt, state_est_prev, z_measurements, config);

        auto [x_post, x_updates, gated_measurements, z_inside_meas,
              existence_prob_upd] =
            update(sens_mod, x_pred, z_pred, z_measurements,
                   existence_prob_pred, config);

        // clang-format off
        return {
            .state = {
                .x_estimate            = x_post,
                .existence_probability = existence_prob_upd,
            },
            .x_prediction       = x_pred,
            .z_prediction       = z_pred,
            .x_updates          = x_updates,
            .gated_measurements = gated_measurements
        };
        // clang-format on
    }

    /**
     * @brief Calculates the predicted existence probability
     * @param existence_prob_est (r_{k-1}) The previous existence probability.
     * @param prob_of_survival (P_s) The probability of survival.
     * @return The predicted existence probability (r_{k|k-1}).
     */
    static double existence_prediction(double existence_prob_est,
                                       double prob_of_survival) {
        double r_km1 = existence_prob_est;
        double P_s = prob_of_survival;
        return P_s * r_km1;  // (7.28)
    }

    /**
     * @brief Calculates the existence probability given the likelihood of the
     * measurements and the previous existence probability.
     * @param z_likelyhoods (l_a_k) The likelihood of the measurements
     * @param existence_prob_est (r_{k-1}) The previous existence probability.
     * @param config The configuration for the IPDA.
     * @return The existence probability (r_k).
     */
    static double existence_prob_update(const Eigen::ArrayXd z_likelyhoods,
                                        double existence_prob_pred,
                                        Config config) {
        double r_kgkm1 = existence_prob_pred;  // r_k given k minus 1
        double P_d = config.pdaf.prob_of_detection;
        double lambda = config.pdaf.clutter_intensity;

        // posterior existence probability r_k
        double L_k = 1 - P_d + P_d / lambda * z_likelyhoods.sum();  // (7.33)
        double r_k = (L_k * r_kgkm1) / (1 - (1 - L_k) * r_kgkm1);   // (7.32)
        return r_k;
    }

    /**
     * @brief Estimates the clutter intensity using (7.31)
     * @param z_pred The predicted measurement.
     * @param existence_prob_pred (r_{k|k-1})  The predicted
     * existence probability.
     * @param num_measurements (m_k) The number of z_measurements.
     * @param config The configuration for the IPDA.
     * @return The clutter intensity.
     */
    static double estimate_clutter_intensity(const Gauss_z& z_pred,
                                             double existence_prob_pred,
                                             double num_measurements,
                                             Config config) {
        size_t m_k = num_measurements;
        double P_d = config.pdaf.prob_of_detection;
        double r_k = existence_prob_pred;
        double V_k =
            utils::Ellipsoid<N_DIM_z>(z_pred, config.pdaf.mahalanobis_threshold)
                .volume();  // gate area

        if (m_k == 0) {
            return 0.0;
        }
        return 1 / V_k * (m_k - r_k * P_d);  // (7.31)
    }
};
}  // namespace vortex::filter
