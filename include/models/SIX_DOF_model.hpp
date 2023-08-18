#pragma once
#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>
namespace Models {

template <int N_3D_VECS, int N_QUATS, int N_INPUTS, int N_MEAS>
class SIX_DOF_model : public Model_base<3 * N_3D_VECS + 4 * N_QUATS, N_MEAS, N_INPUTS, 3 * N_3D_VECS + 3 * N_QUATS, N_INPUTS> {
public:
	static constexpr int n_x = 3 * N_3D_VECS + 4 * N_QUATS;
	static constexpr int n_y = N_MEAS;
	static constexpr int n_u = N_INPUTS;
	static constexpr int n_v = 3 * N_3D_VECS + 3 * N_QUATS;
	static constexpr int n_w = N_INPUTS;

private:
	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)
	using Quaternion = Eigen::Quaternion<double>;
	// defines for position, velocity, angular velocity etc.
	using SpatialVector  = Eigen::Vector3d;
	using RotationVector = Eigen::Vector3d;

	static constexpr int SPATIAL_START = 0;
	static constexpr int QUAT_START    = 3 * N_3D_VECS;

public:
	/**
	 * @brief Six degrees of freedom model that assumes additive noise in spatial states, rotation angles and measurement.
	 * Number of spatial and rotational states can be specified
	 */
	SIX_DOF_model() : Model_base<n_x, n_y, n_u, n_v, n_w>(){};
	~SIX_DOF_model(){};

	/**
	 * @brief Time update function f
	 *
	 * @param Ts Time-step
	 * @param x State
	 * @param u Input
	 * @param v Disturbance
	 * @return State update
	 */
	State f(Time T, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const override final
	{
		(void)Ts;
		(void)x;
		(void)u;
		(void)v;
		State x_dot;
		return x_dot;
	}
	virtual State f(Time T, const State &x, const Input &u = Input::Zero()) const = 0;

	/**
	 * @brief Measurement function h
	 *
	 * @param Ts Time-step
	 * @param x State
	 * @param w Noise
	 * @return Measurement
	 */
	Measurement h(Time T, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const override final
	{
		(void)T;
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
	Eigen::Map<SpatialVector> spatial(State &x, int index) const { return Eigen::Map<SpatialVector>(x.data() + SPATIAL_START + index * 3, 3); }

	/**
	 * @brief Get map to quaternion states (4xN_QUATS) matrix
	 * @param x State
	 * @return Map to quaternion states
	 */
	Eigen::Map<Quaternion> quaternion(State &x, int index) const { return Eigen::Map<Quaternion>(x.data() + QUAT_START + index * 4); }

	/**
	 * @brief Rodrigues formula for quaternion multiplication
	 * @param q Quaternion
	 * @param w Angular velocity
	 * @return Quaternion derivative
	 */
	Quaternion diff_quaternion(const Quaternion &q, const RotationVector &w) const
	{
		Quaternion q_dot;
		q_dot.w()   = -0.5 * w.dot(q.vec());
		q_dot.vec() = 0.5 * (w * q.w() + q.vec().cross(w));
		return q_dot;
	}

	/**
	 * @brief Quaternion to rotation matrix
	 *
	 * @param q Quaternion
	 * @return Rotation matrix
	 */
	Eigen::Matrix3d quaternion_to_rotation_matrix(const Quaternion &q) const
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
	Quaternion rotation_matrix_to_quaternion(const Eigen::Matrix3d &R) const
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
	Quaternion rotation_vector_to_quaternion(const RotationVector &rot_vec) const
	{
		double theta = rot_vec.norm();
		// if angle is zero, return identity quaternion
		if (theta < 1e-6) {
			return Quaternion::Identity();
		}
		// else compute quaternion
		Quaternion q;
		q.w()   = cos(theta / 2.0);
		q.vec() = sin(theta / 2.0) * rot_vec / theta;

		return q;
	}
};
} // namespace Models