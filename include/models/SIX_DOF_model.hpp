#pragma once
#include <models/model_definitions.hpp>
#include <models/Model_base.hpp>
namespace Models {

template<int N_3D_VECS, int N_QUATS, int N_INPUTS, int N_MEAS>
class SIX_DOF_model : public Model_base<3*N_3D_VECS+4*N_QUATS, N_MEAS, N_INPUTS, 3*N_3D_VECS+3*N_QUATS, N_INPUTS> {
    using n_x = 3*N_3D_VECS+4*N_QUATS;
    using n_y = N_MEAS;
    using n_u = N_INPUTS;
    using n_v = 3*N_3D_VECS+3*N_QUATS;
    using n_w = N_INPUTS;

    DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
    using Quaternion = Eigen::Quaternion<double>;
    // defines for position, velocity, angular velocity etc.
    using SpatialVector = Eigen::Vector3d;
    using RotationVector = Eigen::Vector3d;

    using SPATIAL_START = 0;
    using QUAT_START = 3*N_3D_VECS;

public:
    /**
     * @brief Six degrees of freedom model that assumes additive noise in spatial states, rotation angles and measurement.
     * Number of spatial and rotational states can be specified
    */
    SIX_DOF_model() : Model_base<n_x,n_y,n_u,n_v,n_w>() {};
    ~SIX_DOF_model() {};

    /**
     * @brief Time update function f
     * 
     * @param Ts Time-step
     * @param x State
     * @param u Input
     * @param v Disturbance
     * @return State update
     */
    State f(Timestep Ts, State x, Input u = Input::Zero(), Disturbance v = Disturbance::Zero()) override final
    {
        (void)Ts;
        (void)x;
        (void)u;
        (void)v;
        State x_dot;
        return x_dot;
    }

    /**
     * @brief Measurement function h
     * 
     * @param Ts Time-step
     * @param x State
     * @param w Noise
     * @return Measurement
     */
    Measurement h(Timestep Ts, State x, Input u = Input::Zero(), Noise w = Noise::Zero()) override final
    {
        (void)Ts;
        (void)x;
        (void)w;
        Measurement y;
        return y;
    }

    /**
     * @brief Get map to spatial states (3xN_3D_VECS) matrix
     * @param x State
     * @return Map to spatial states
     */
    Eigen::Map<SpatialVector> spatial(State x, int index)
    {
        return Eigen::Map<SpatialVector>(x.data() + SPATIAL_START + index*3, 3);
    }

    /**
     * @brief Get map to quaternion states (4xN_QUATS) matrix
     * @param x State
     * @return Map to quaternion states
     */
    Eigen::Map<Quaternion> quaternion(State x, int index)
    {
        return Eigen::Map<Quaternion>(x.data() + QUAT_START + index*4);
    }


    /**
     * @brief Rodrigues formula for quaternion multiplication
     * @param q Quaternion
     * @param w Angular velocity
     * @return Quaternion derivative    
     */
    Quaternion diff_quaternion(Quaternion q, RotationVector w)
    {
        Quaternion q_dot;
        q_dot.w() = -0.5 * w.dot(q.vec());
        q_dot.vec() = 0.5 * (w * q.w() + q.vec().cross(w));
        return q_dot;
    }

    /**
     * @brief Quaternion to rotation matrix
     * 
     * @param q Quaternion
     * @return Rotation matrix
     */
    Eigen::Matrix3d quaternion_to_rotation_matrix(Quaternion q)
    {
        Eigen::Matrix3d R;
        R = q.toRotationMatrix();
        return R;
    }

    /**
     * @brief Rotation matrix to quaternion
     * 
     * @param R Rotation matrix
     * @return Quaternion
     */
    Quaternion rotation_matrix_to_quaternion(Eigen::Matrix3d R)
    {
        Quaternion q;
        q = R;
        return q;
    }

    /**
        * @brief Rotation vector to quaternion
        * 
        * @param angle Rotation vector
        * @return Quaternion
    */
    Quaternion rotation_vector_to_quaternion(RotationVector rot_vec)
    {
        double theta = rot_vec.norm();
        // if angle is zero, return identity quaternion
        if (theta < 1e-6)
        {
            return Quaternion::Identity();
        }
        // else compute quaternion
        Quaternion q;
        q.w() = cos(theta / 2.0);
        q.vec() = sin(theta / 2.0) * rot_vec / theta;

        return q;
    }
};
}