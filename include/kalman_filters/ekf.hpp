#pragma once
#include <models.hpp>

class EKF 
{
public:
    EKF(Model::EKF_Dynamics_model* ekf_dynamics_model, Model::EKF_Dynamics_model* ekf_measurement_model)
    : dynamics_model{ekf_dynamics_model}, measurement_model{measurement_model} {}

    // x
    State predicted_state_estimate(Timestep Ts, State x_prev) 
    {
        return dynamics_model->f(Ts,x_prev);
    }

    // P_xx
    Mat predicted_covariance(Timestep Ts, State x, Mat P_prev) 
    {
        Mat F = dynamics_model->F(Ts,x);
        Mat Q = dynamics_model->Q(Ts,x);
        return F*P_prev*F.transpose() + Q;
    }

    // y_est
    Measurement predicted_output(Timestep Ts, State x)
    {
        return measurement_model->h(Ts,x);
    }

    // P_yy
    Mat output_covariance(Timestep Ts, State x, Mat P_xx)
    {
        Mat H = measurement_model->H(Ts,x);
        Mat R = measurement_model->R(Ts,x);
        return H*P_xx*H.transpose() + R;
    }

    // P_xy
    Mat cross_covariance(Timestep Ts, State x, Mat P_xx)
    {
        return P_xx*measurement_model->H(Ts,x);
    }

    // K
    Mat kalman_gain(Mat P_xy, Mat P_yy)
    {
        // Use Cholesky decomposition for inverting P_yy
        size_t m = P_yy.rows();
        Mat I = Eigen::MatrixXf::Identity(m,m);
        Mat P_yy_inv = P_yy.llt().solve(I);
        return P_xy*P_yy_inv;
    }

    //x_k+1
    State corrected_state_estimate(State x_est, Measurement y_meas, Measurement y_est, Mat K)
    {
        return x_est + K*(y_meas-y_est);
    }
    // P_k+1
    Mat corrected_covariance(Timestep Ts, State x, Mat P, Mat K)
    {
        State h = measurement_model->h(Ts,x);
        return (Eigen::MatrixXd::Identity()-K*h)*P;
    }
private:
    Model::EKF_Dynamics_model* dynamics_model;
    Model::EKF_Measurement_model* measurement_model;
};
