#pragma once
#include <models.hpp>

// Calculate jacobians
// x_k- = f(x_k-1, u_k-1, v_k-1)
// P_x_k- =

class EKF 
{
public:
    EKF(Model::EKF_Dynamics_model* dynamics_model, Model::EKF_Dynamics_model* Measurement_model)
    : dynamics_model{dynamics_model}, measurement_model{measurement_model} {}

    State predicted_state_estimate(Timestep Ts, State x) 
    {
        return dynamics_model->f(Ts,x);
    }

    Mat predicted_covariance(Timestep Ts, State x, Mat P) 
    {
        Mat F = dynamics_model->F(Ts,x);
        Mat Q = dynamics_model->Q(Ts,x);
        return F*P*F.transpose() + Q;
    }
    void predicted_output();
    void output_covariance();
    void cross_covariance();
    void kalman_gain();
    void corrected_state_estimate();
    void corrected_covariance();
private:
    Model::EKF_Dynamics_model* dynamics_model;
    Model::EKF_Dynamics_model* measurement_model;
};
