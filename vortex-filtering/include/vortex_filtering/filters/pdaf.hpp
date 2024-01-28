# pragma once

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

namespace vortex {
namespace filter {

using std::string;
// using std::endl;

template<class DynModT, class SensModT>
class PDAF {
public:
    using SensModI = typename SensModT::SensModI;
    using DynModI = typename DynModT::DynModI;
    using DynModPtr = std::shared_ptr<DynModI>;
    using SensModPtr = std::shared_ptr<SensModI>;
    using EKF = vortex::filter::EKF<DynModI, SensModI>;
    using Gauss_z = typename SensModI::Gauss_z;
    using Gauss_x = typename DynModI::Gauss_x;
    using Vec_z = typename SensModI::Vec_z;
    using MeasurementsZd = std::vector<Vec_z>;
    using StatesXd = std::vector<Gauss_x>;
    using GaussMixZd = vortex::prob::GaussianMixture<DynModI::N_DIM_x>;

    PDAF()
    {
        std::cout << "Created PDAF class!" << std::endl;
    }

    std::tuple<Gauss_x, MeasurementsZd, MeasurementsZd, Gauss_x, Gauss_z, StatesXd> predict_next_state(
        const Gauss_x& x_est, 
        const MeasurementsZd& z_meas, 
        double timestep, 
        const DynModPtr& dyn_model, 
        const SensModPtr& sen_model,
        double gate_threshold,
        double prob_of_detection,
        double clutter_intensity
        ) const
    {
        auto [x_pred, z_pred] = EKF::predict(dyn_model, sen_model, timestep, x_est);
        auto [inside, outside] = apply_gate(z_meas, z_pred, gate_threshold);

        StatesXd x_updated;
        for (const auto& measurement : inside) {
            x_updated.push_back(EKF::update(sen_model, x_pred, z_pred, measurement));
        }

        Gauss_x x_final = get_weighted_average(
            z_meas,
            x_updated, 
            z_pred, 
            x_pred, 
            prob_of_detection, 
            clutter_intensity);
        return {x_final, inside, outside, x_pred, z_pred, x_updated};
    }

    std::tuple<MeasurementsZd, MeasurementsZd> apply_gate(
        const MeasurementsZd& z_meas, 
        const Gauss_z& z_pred,
        double gate_threshold
        ) const
    {
        MeasurementsZd inside_meas;
        MeasurementsZd outside_meas;

        for (const auto& measurement : z_meas) {
            double distance = z_pred.mahalanobis_distance(measurement);
            std::cout << "measurement: " << measurement << std::endl;
            std::cout << "z_pred: " << z_pred.mean() << std::endl;
            std::cout << "distance: " << distance << std::endl;

            if (distance <= gate_threshold) {
                inside_meas.push_back(measurement);
            } else {
                outside_meas.push_back(measurement);
            }
        }

        return {inside_meas, outside_meas};
    }

    // Getting weighted average of the predicted states
    Gauss_x get_weighted_average(
        const MeasurementsZd& z_meas,
        const StatesXd& updated_states,
        const Gauss_z& z_pred,
        const Gauss_x& x_pred,
        double prob_of_detection,
        double clutter_intensity
        ) const
    {
        StatesXd states;
        states.push_back(x_pred);
        states.insert(states.end(), updated_states.begin(), updated_states.end());

        Eigen::VectorXd weights = get_weights(z_meas, z_pred, prob_of_detection, clutter_intensity);

        GaussMixZd gaussian_mixture(weights, states);

        return gaussian_mixture.reduce();
    }

    // Getting association probabilities according to textbook p. 123 "Corollary 7.3.3"
    Eigen::VectorXd get_weights(
        const MeasurementsZd& z_meas,
        const Gauss_z& z_pred,
        double prob_of_detection,
        double clutter_intensity
        ) const
    {
        Eigen::VectorXd weights(z_meas.size() + 1);

        // in case no measurement assosiates with the target
        double no_association = clutter_intensity * (1 - prob_of_detection);
        weights(0) = no_association;

        // measurements associating with the target
        for (size_t k = 1; k < z_meas.size() + 1; k++) {
            weights(k) = (prob_of_detection * z_pred.pdf(z_meas.at(k - 1)));
        }

        // normalize weights
        weights /= weights.sum();

        return weights;
    }
};

}  // namespace filter
}  // namespace vortex