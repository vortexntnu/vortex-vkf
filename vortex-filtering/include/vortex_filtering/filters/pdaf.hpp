#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

using std::string;
// using std::endl;

template<class DynModT, class SensModT>
class PDAF {
public:
    using SensModI = typename SensModT::SensModI;
    using DynModI = typename DynModT::BaseI;
    using DynModPtr = std::shared_ptr<DynModI>;
    using SensModPtr = std::shared_ptr<SensModI>;
    using EKF = vortex::filter::EKF_M<DynModI, SensModI>;
    using Gauss_z = typename SensModI::Gauss_z;
    using Gauss_x = typename DynModI::Gauss_x;
    using Vec_z = typename SensModI::Vec_z;
    using MeasurementsZd = std::vector<Vec_z>;
    using StatesXd = std::vector<Gauss_x>;
    using GaussMixZd = vortex::prob::GaussianMixture<DynModI::N_DIM_x>;

    double gate_threshold_;
    double prob_of_detection_;
    double clutter_intensity_;

    PDAF(double gate, double prob_of_detection, double clutter_intensity)
        : gate_threshold_(gate)
        , prob_of_detection_(prob_of_detection)
        , clutter_intensity_(clutter_intensity)
    {
        std::cout << "Created PDAF class with given models!" << std::endl;
    }

    std::tuple<Gauss_z, MeasurementsZd, MeasurementsZd> predict_next_state(Gauss_x x_est, MeasurementsZd z_meas, double timestep, DynModPtr dyn_model, SensModPtr sen_model)
    {
        EKF ekf;
        auto [x_pred, z_pred] = ekf.predict(dyn_model, sen_model, timestep, x_est);
        auto [inside, outside] = apply_gate(gate_threshold_, z_meas, z_pred);

        StatesXd updated;
        for (const auto& measurement : inside) {
            updated.push_back(ekf.update(x_pred, z_pred, measurement));
        }

        Gauss_z predicted_state = get_weighted_average(z_meas, updated, z_pred, x_pred);
        return {predicted_state, inside, outside};
    }

    std::tuple<MeasurementsZd, MeasurementsZd> apply_gate(double gate_threshold, MeasurementsZd z_meas, Gauss_z z_pred)
    {
        MeasurementsZd inside_meas;
        MeasurementsZd outside_meas;

        for (const auto& measurement : z_meas) {
            double distance = z_pred.mahalanobis_distance(measurement);

            if (distance < gate_threshold) {
                inside_meas.push_back(measurement);
            } else {
                outside_meas.push_back(measurement);
            }
        }

        return {inside_meas, outside_meas};
    }

    // Getting weighted average of the predicted states
    Gauss_x get_weighted_average(MeasurementsZd z_meas, StatesXd updated_states, Gauss_z z_pred, Gauss_x x_pred)
    {
        StatesXd states;
        states.push_back(x_pred);
        states.insert(states.end(), updated_states.begin(), updated_states.end());

        Eigen::VectorXd weights = get_weights(z_meas, z_pred);

        GaussMixZd gaussian_mixture(weights, states);

        return gaussian_mixture.reduce();
    }

    // Getting association probabilities according to textbook p. 123 "Corollary 7.3.3"
    Eigen::VectorXd get_weights(MeasurementsZd z_meas, Gauss_z z_pred)
    {
        Eigen::VectorXd weights(z_meas.size() + 1);

        // in case no measurement assosiates with the target
        double no_association = clutter_intensity_ * (1 - prob_of_detection_);
        weights(0) = no_association;

        // measurements associating with the target
        for (size_t k = 1; k < z_meas.size() + 1; k++) {
            weights(k) = (prob_of_detection_ * z_pred.pdf(z_meas.at(k - 1)));
        }

        // normalize weights
        weights /= weights.sum();

        return weights;
    }

    // Getter and setter for mem
    void set_gate_threshold(double gate_threshold)
    {
        gate_threshold_ = gate_threshold;
    }

    void set_prob_of_detection(double prob_of_detection)
    {
        prob_of_detection_ = prob_of_detection;
    }   

    void set_clutter_intensity(double clutter_intensity)
    {
        clutter_intensity_ = clutter_intensity;
    }   

    double get_gate_threshold()
    {
        return gate_threshold_;
    }   

    double get_prob_of_detection()
    {
        return prob_of_detection_;
    }   

    double get_clutter_intensity()
    {
        return clutter_intensity_;
    } 
};