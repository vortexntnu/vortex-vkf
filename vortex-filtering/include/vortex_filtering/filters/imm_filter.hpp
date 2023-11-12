/**
 * @file imm_filter.hpp
 * @author Eirik Kolås
 * @brief IMM filter based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-11-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/sensor_model.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/filters/ekf.hpp>

namespace vortex {
namespace filters {

template <typename ImmModelT, class SensModT>
class ImmFilter {
public:
    using ImmModelT::N_DIM_x;
    using ImmModelT::N_MODELS;
    using ImmModelT::Mat_nn;
    using ImmModelT::Vec_n;
    using ImmModelT::Vec_x;
    using ImmModelT::Mat_xx;
    using ImmModelT::Vec_x;
    using SensModT::N_DIM_z;
    using SensModT::Mat_zz;
    using SensModT::Vec_z;
    using SensModT::Mat_xz;


    ImmFilter(const ImmModelT& imm_model, const SensModT& sensor_model)
        : imm_model_(imm_model), sensor_model_(sensor_model) 
        {
            for (int i = 0; i < N_MODELS; i++) {
                ekfs_.push_back(std::make_unique<EKF<DynamicModel, SensorModel>>(imm_model_.get_dynamic_models().at(i), sensor_model_));
            }
        }

    /**
     * @brief Calculate mixing probabilities, following step 1 in (6.4.1) in the book
     * 
     * @param x_est_prev Mixture from previous time step
     * @param dt 
     */
    calculate_mixings(const prob::GaussianMixture<N_DIM_x>& x_est_prev, double dt) 
    {
        Mat_nn pi_mat = imm_model_.get_pi_mat_d(dt);
        Vec_nn prev_weights = Eigen::Map<Vec_nn>(x_est_prev.weights().data(), weight_vec.size()); // Convert to Eigen vector

        // Mat_nn mixing_probs = pi_mat.cwiseProduct(prev_weights.transpose().replicate(N_MODELS, 1)); // Multiply each column with the corresponding weight
        for (int i = 0; i < mixing_probs.rows(); i++) {
        // TODO: Check if this is correct
            mixing_probs.row(i) = pi_mat.row(i).cwiseProduct(prev_weights.transpose());
        // Normalize
            mixing_probs.row(i) /= mixing_probs.row(i).sum();
        }
        return mixing_probs;
    }

    /**
     * @brief Calculate moment-based approximation, following step 2 in (6.4.1) in the book
     * @param x_est_prev Mixture from previous time step
     * @param mixing_probs Mixing probabilities
     */
    std::vector<prob::GaussianMixture<N_DIM_x>> mixing(const prob::GaussianMixture<N_DIM_x>& x_est_prev, Mat_nn mixing_probs) 
    {
        std::vector<prob::MultiVarGauss<N_DIM_x>> moment_based_preds(N_MODELS);
        for (int i = 0; i < N_MODELS; i++) {
            prob::GaussianMixture<N_DIM_x> mixture(mixing_probs.row(i), x_est_prev.gaussians());
            moment_based_preds.append(mixture.reduce());
        }
        return moment_based_preds;
    }

    /**
     * @brief Calculate the mode-match filter outputs (6.36), following step 3 in (6.4.1) in the book
     * 
     * @param moment_based_preds Moment-based predictions
     * @param z_meas Vec_z
     * @param dt Time step
     */
    std::tuple<std::vector<prob::MultiVarGauss<N_DIM_x>>, std::vector<prob::MultiVarGauss<N_DIM_x>>, std::vector<prob::MultiVarGauss<N_DIM_z>>>> mode_match_filter(const std::vector<prob::MultiVarGauss<N_DIM_x>>& moment_based_preds, Vec_zz z_meas, double dt) 
    {
        
        std::tuple<std::vector<prob::MultiVarGauss<N_DIM_x>>, std::vector<prob::MultiVarGauss<N_DIM_x>>, std::vector<prob::MultiVarGauss<N_DIM_z>>>> ekf_outs;

        for (int i = 0; i < N_MODELS; i++) {
            auto [x_est_upd, x_est_pred, z_est_pred] = ekfs_.at(i).step(moment_based_preds.at(i), z_meas, dt);
            ekf_outs<0>.append(x_est_upd);
            ekf_outs<1>.append(x_est_pred);
            ekf_outs<2>.append(z_est_pred);
        }
        return ekf_outs;
    }

    /**
     * @brief Update the mixing probabilites using (6.37) from setp 3 and (6.38) from step 4 in (6.4.1) in the book
     * @param ekf_outs Mode-match filter outputs
     * @param z_meas Vec_z
     * @param dt Time step
     * @param weights Weights 
     */
    Mat_nn update_probabilities(const std::vector<std::tuple<ImmModelT, SensModT>>& ekf_outs, Vec_z z_meas, double dt, Vec_n weights) 
    {
        Mat_nn pi_mat = imm_model_.get_pi_mat_d(dt);
        Vec_n weights_pred = pi_mat.transpose() * weights

        Vec_n z_probs = Vec_n::Zero();
        for (int i = 0; i < N_MODELS; i++) {
            auto [x_est_upd, z_est_pred] = ekf_outs.at(i);
            z_probs(i) = z_est_pred.pdf(z_meas);
        }

        Vec_n weights_upd = z_probs.cwiseProduct(weights_pred);
        weights_upd /= weights_upd.sum();

        return weights_upd;

    }

    /**
     * @brief Perform one IMM filter step
     * 
     * @param x_est_prev Mixture from previous time step
     * @param z_meas Vec_z
     * @param dt Time step
     */
    std::tuple<prob::GaussianMixture<N_DIM_x>, prob::GaussianMixture<N_DIM_x>, prob::GaussianMixture<N_DIM_z> step(const prob::GaussianMixture<N_DIM_x>& x_est_prev, Vec_z z_meas, double dt)
    {
        Mat_nn mixing_probs = calculate_mixings(x_est_prev, dt);
        std::vector<prob::MultiVarGauss<N_DIM_x>> moment_based_preds = mixing(x_est_prev, mixing_probs);
        auto [x_est_upds, x_est_preds, z_est_preds] = mode_match_filter(moment_based_preds, z_meas, dt);
        Vec_n weights_upd = update_probabilities(ekf_outs, z_meas, dt, mixing_probs.row(0));

        prob::GaussianMixture<N_DIM_x> x_est_upd{weights_upd, x_est_upds};
        prob::GaussianMixture<N_DIM_x> x_est_pred(x_est_prev.weights, x_est_preds);
        prob::GaussianMixture<N_DIM_z> z_est_pred(x_est_prev.weights, z_est_preds);

        return {x_est_upd, x_est_pred, z_est_pred};
    }

private:
    ImmModel imm_model_;
    SensModT sensor_model_;
    std::vector<std::unique_ptr<EKF<ImmModelT, SensModT>>> ekfs_;
};

} // namespace filters
} // namespace vortex