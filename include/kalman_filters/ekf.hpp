#pragma once
#include <models.hpp>

// Calculate jacobians
// x_k- = f(x_k-1, u_k-1, v_k-1)
// P_x_k- =

class EKF 
{
public:
    EKF(model::LTV_Model dynamics_model, model::LTV_model measurement_model)
    : dynamics_model{dynamics_model}, measurement_model{measurement_model} {}

    void predicted_state_estimate(Timestep Ts, State x) {return dynamics_model.f(Ts,x)}
    void predicted_covariance() {}
    void predicted_output();
    void output_covariance();
    void cross_covariance();
    void kalman_gain();
    void corrected_state_estimate();
    void corrected_covariance();
private:
    model::LTV_model dynamics_model;
    model::LTV_model measurement_model;
};
