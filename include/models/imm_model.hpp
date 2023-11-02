/**
 * @file imm_model.hpp
 * @author Eirik Kolås 
 * @brief 
 * @version 0.1
 * @date 2023-11-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <models/dynamic_model.hpp>
#include <Eigen/Dense>
#include <memory>

namespace vortex {
namespace models {

template <int n_dim_x, int n_models>
class ImmModel : public DynamicModel<n_dim_x> {
public:
    using typename DynamicModel<n_dim_x>::State;
    using typename DynamicModel<n_dim_x>::Mat_xx;
    using Vec_nn = Eigen::Vector<double, n_models>;
    using Mat_nn = Eigen::Matrix<double, n_models, n_models>;

    /**
     * @brief Construct a new Imm Model object
     * @param models Models to use
     * @param jump_matrix Markov jump chain matrix for the transition probabilities. 
     * I.e. the probability of switching from model i to model j is jump_matrix(i,j). Diagonal should be 0.
     * @param hold_times Holding time for each state. Parameter is the mean of an exponential distribution.
     */
    ImmModel(std::vector<std::shared_ptr<DynamicModel<n_dim_x>>> models, Mat_nn jump_matrix, Vec_nn hold_times)
        : models_(std::move(models)), jump_matrix_(std::move(jump_matrix)), hold_times_(std::move(hold_times)), N_MODELS_(n_models)
    {
        // Validate input
        assert(models_.size() == N_MODELS_);
        assert(hold_times_.size() == N_MODELS_);
        assert(jump_matrix_.diagonal().isZero());

        // Normalize jump matrix
        for (int i = 0; i < jump_matrix_.rows(); i++) {
            jump_matrix_.row(i) /= jump_matrix_.row(i).sum();
        }
    }

    /**
     * @brief Continuous time dynamics for the first model in the list.
     * Use f_c(x, model_idx) to get the dynamics of a specific model.
     * @param x State
     * @return The first model in the list. 
     */
    State f_c(const State& x) const override
    {
        // error if used
        assert(false);
        return models_.at(0)->f_c(x);
    }

    /**
     * @brief Continuous time dynamics for a specific model
     * 
     * @param x 
     * @param model_idx 
     * @return State 
     */
    State f_c(const State& x, int model_idx) const
    {
        return models_.at(model_idx)->f_c(x);
    }

    /**
     * @brief Compute the continuous-time transition matrix
     * See https://en.wikipedia.org/wiki/Continuous-time_Markov_chain
     * @return Matrix Continuous-time transition matrix
     */
    Mat_nn get_pi_mat_c() const
    {
        static bool is_cached = false;
        static Mat_nn pi_mat_c = Mat_nn::Zero();
        if (is_cached) { return pi_mat_c; }

        Vec_nn t_inv = hold_times_.cwiseInverse();
        Mat_nn = - t_inv.asDiagonal() + t_inv*jump_matrix_;

        is_cached = true;
        return pi_mat_c;
    }

    /**
     * @brief Compute the discrete-time transition matrix
     * @return Matrix Discrete-time transition matrix
     */
     */
    Mat_nn get_pi_mat_d(double dt) const
    {
        return (get_pi_mat_c() * dt).exp();
    }

    /**
     * @brief Get the dynamic models
     * @return models
     */
    std::vector<std::shared_ptr<DynamicModel<n_dim_x>>> get_models() const
    {
        return models_;
    }
private:
    const std::vector<std::shared_ptr<DynamicModel<n_dim_x>>> models_;
    const Mat_nn jump_matrix_;
    const Vec_nn hold_times_;
public:
    const size_t N_MODELS_;
};
} // namespace models
} // namespace vortex