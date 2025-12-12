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
    double min_gate_threshold = 0.0;
    double max_gate_threshold = std::numeric_limits<double>::max();
    double prob_of_detection = 1.0;
    double clutter_intensity = 1.0;
};
}  // namespace config

template <concepts::DynamicModelLTVWithDefinedSizes DynModT,
          concepts::SensorModelLTVWithDefinedSizes SensModT>
class PDAF {
   public:
    static constexpr int N_DIM_x = DynModT::N_DIM_x;
    static constexpr int N_DIM_z = SensModT::N_DIM_z;
    static constexpr int N_DIM_u = DynModT::N_DIM_u;
    static constexpr int N_DIM_v = DynModT::N_DIM_v;
    static constexpr int N_DIM_w = SensModT::N_DIM_w;

    using T = Types_xzuvw<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

    using Gauss_z = typename T::Gauss_z;
    using Gauss_x = typename T::Gauss_x;
    using Vec_z = typename T::Vec_z;
    using GaussMix_x = typename T::GaussMix_x;
    using GaussMix_z = typename T::GaussMix_z;
    using Arr_zXd = Eigen::Array<double, N_DIM_z, Eigen::Dynamic>;
    using Arr_1Xb = Eigen::Array<bool, 1, Eigen::Dynamic>;
    using Gauss_xX = std::vector<Gauss_x>;
    using EKF =
        vortex::filter::EKF_t<N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w>;

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
     * @brief Perform one PDAF prediction step
     *
     * @param dyn_mod The dynamic model
     * @param sens_mod The sensor model
     * @param dt Time step in seconds
     * @param x_est The estimated state
     * @return `std::pair<Gauss_x, Gauss_z>` The predicted state and predicted
     * measurement
     */
    static std::pair<Gauss_x, Gauss_z> predict(const DynModT& dyn_mod,
                                               const SensModT& sens_mod,
                                               double dt,
                                               const Gauss_x& x_est) {
        auto [x_pred, z_pred] = EKF::predict(dyn_mod, sens_mod, dt, x_est);

        return {x_pred, z_pred};
    }

    /**
     * @brief Perform one PDAF update step
     *
     * @param sens_mod The sensor model
     * @param x_pred The predicted state
     * @param z_pred The predicted measurement
     * @param z_measurements Array of measurements
     * @param config Configuration for the PDAF
     * @return `std::tuple<Gauss_x, Gauss_xX, Arr_1Xb>` The updated state, all
     * updated states for each measurement, and the indices of the measurements
     * that are inside the gate
     */
    static std::tuple<Gauss_x, Gauss_xX, Arr_1Xb, Arr_zXd> update(
        const SensModT& sens_mod,
        const Gauss_x& x_pred,
        const Gauss_z& z_pred,
        const Arr_zXd& z_measurements,
        const Config& config) {
        auto gated_measurements = apply_gate(z_measurements, z_pred, config);
        auto z_inside_meas =
            get_inside_measurements(z_measurements, gated_measurements);

        Gauss_xX x_updates;
        for (const auto& z_k : z_inside_meas.colwise()) {
            x_updates.push_back(EKF::update(sens_mod, x_pred, z_pred, z_k));
        }

        Gauss_x x_post = get_weighted_average(
            z_inside_meas, x_updates, z_pred, x_pred,
            config.pdaf.prob_of_detection, config.pdaf.clutter_intensity);

        return {x_post, x_updates, gated_measurements, z_inside_meas};
    }

    /**
     * @brief Perform one step of the Probabilistic Data Association Filter
     *
     * @param dyn_mod The dynamic model
     * @param sens_mod The sensor model
     * @param dt Time step in seconds
     * @param x_est The estimated state
     * @param z_measurements Array of measurements
     * @param config Configuration for the PDAF
     * @return `Output` The result of the PDAF step and some intermediate
     * results
     */
    static Output step(const DynModT& dyn_mod,
                       const SensModT& sens_mod,
                       double dt,
                       const Gauss_x& x_est,
                       const Arr_zXd& z_measurements,
                       const Config& config) {
        auto [x_pred, z_pred] = predict(dyn_mod, sens_mod, dt, x_est);

        auto [x_final, x_updates, gated_measurements, z_inside_meas] =
            update(sens_mod, x_pred, z_pred, z_measurements, config);
        return {x_final, x_pred, z_pred, x_updates, gated_measurements};
    }

    /**
     * @brief Apply gate to the measurements
     *
     * @param z_measurements Array of measurements
     * @param z_pred Predicted measurement
     * @param config Configuration for the PDAF
     * @return `Arr_1Xb` Indices of the measurements that are inside the gate
     */
    static Arr_1Xb apply_gate(const Arr_zXd& z_measurements,
                              const Gauss_z& z_pred,
                              Config config) {
        double mahalanobis_threshold = config.pdaf.mahalanobis_threshold;
        double min_gate_threshold = config.pdaf.min_gate_threshold;
        double max_gate_threshold = config.pdaf.max_gate_threshold;

        Arr_1Xb gated_measurements(1, z_measurements.cols());

        for (size_t a_k = 0; const Vec_z& z_k : z_measurements.colwise()) {
            double mahalanobis_distance = z_pred.mahalanobis_distance(z_k);
            double regular_distance = (z_pred.mean() - z_k).norm();
            gated_measurements(a_k++) =
                (mahalanobis_distance <= mahalanobis_threshold ||
                 regular_distance <= min_gate_threshold) &&
                regular_distance <= max_gate_threshold;
        }
        return gated_measurements;
    }

    /**
     * @brief Get the measurements that are inside the gate
     *
     * @param z_measurements Array of measurements
     * @param gated_measurements Indices of the measurements that are inside the
     * gate
     * @return `Arr_zXd` The measurements that are inside the gate
     */
    static Arr_zXd get_inside_measurements(const Arr_zXd& z_measurements,
                                           const Arr_1Xb& gated_measurements) {
        Arr_zXd z_inside_meas(N_DIM_z, gated_measurements.count());
        for (size_t i = 0, j = 0; bool gated : gated_measurements) {
            if (gated) {
                z_inside_meas.col(j++) = z_measurements.col(i);
            }
            i++;
        }
        return z_inside_meas;
    }

    /**
     * @brief Get the measurement likelihoods
     *
     * @param z_pred The predicted measurement
     * @param z_measurements The measurements
     *
     * @return `Eigen::ArrayXd` The measurement likelihoods
     */
    static Eigen::ArrayXd get_measurement_likelihoods(
        const Gauss_z& z_pred,
        const Arr_zXd& z_measurements) {
        const int m = z_measurements.cols();
        Eigen::ArrayXd meas_likelihoods(m);

        int i = 0;
        for (const Vec_z& z_k : z_measurements.colwise()) {
            meas_likelihoods(i++) = z_pred.pdf(z_k);
        }

        return meas_likelihoods;
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
    static Gauss_x get_weighted_average(const Arr_zXd& z_measurements,
                                        const Gauss_xX& updated_states,
                                        const Gauss_z& z_pred,
                                        const Gauss_x& x_pred,
                                        double prob_of_detection,
                                        double clutter_intensity) {
        Gauss_xX states;
        states.push_back(x_pred);
        states.insert(states.end(), updated_states.begin(),
                      updated_states.end());

        Eigen::VectorXd weights = get_weights(
            z_measurements, z_pred, prob_of_detection, clutter_intensity);

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
    static Eigen::VectorXd get_weights(const Arr_zXd& z_measurements,
                                       const Gauss_z& z_pred,
                                       double prob_of_detection,
                                       double clutter_intensity) {
        double lambda = clutter_intensity;
        double P_d = prob_of_detection;

        Eigen::VectorXd weights(z_measurements.cols() + 1);

        // in case no measurement associates with the target
        if (lambda == 0.0 && z_measurements.cols() == 0) {
            weights(0) = 1.0;
        } else {
            weights(0) = lambda * (1 - P_d);
        }

        // measurements associating with the target
        for (size_t a_k = 1; const Vec_z& z_k : z_measurements.colwise()) {
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
    static Eigen::ArrayXd association_probabilities(
        const Eigen::ArrayXd& z_likelyhoods,
        double prob_of_detection,
        double clutter_intensity) {
        size_t m_k = z_likelyhoods.size();
        double lambda = clutter_intensity;
        double P_d = prob_of_detection;

        Eigen::ArrayXd weights(m_k + 1);

        // Association probabilities (Corrolary 7.3.3)
        weights(0) = lambda * (1 - P_d);
        weights.tail(m_k) = P_d * z_likelyhoods;

        // normalize weights
        weights /= weights.sum();

        return weights;
    }
};

}  // namespace vortex::filter
