#pragma once
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

namespace vortex {
namespace models {

/**
 * A simple sensor model for testing purposes.
 * The measurement model is simply the n_dim_z first elements of the state vector.
 * @tparam n_dim_x Dimension of state
 * @tparam n_dim_z Dimension of measurement
 */
template <int n_dim_x, int n_dim_z> class IdentitySensorModel : public interface::SensorModelLTV<n_dim_x, n_dim_z> {
public:
	using SensModI = interface::SensorModelI<n_dim_x, n_dim_z>;

	using typename SensModI::Mat_xx;
	using typename SensModI::Mat_zx;
	using typename SensModI::Mat_zz;
	using typename SensModI::Vec_x;
	using typename SensModI::Vec_z;

	/** Construct a new Simple Sensor Model object.
	 * The measurement model is simply the n_dim_z first elements of the state vector.
	 * @param std Standard deviation. Sets the measurement covariance matrix R to I*std^2.
	 * @tparam n_dim_x Dimension of state
	 * @tparam n_dim_z Dimension of measurement
	 */
	IdentitySensorModel(double std) : R_(Mat_zz::Identity() * std * std) {}

	/** Construct a new Simple Sensor Model object.
	 * The measurement model is simply the n_dim_z first elements of the state vector.
	 * @param R Measurement covariance matrix
	 */
	IdentitySensorModel(Mat_zz R) : R_(R) {}

	/** Get the Jacobian of the measurement model with respect to the state.
	 * @param x State (not used)
	 * @return Mat_zx
	 * @note Overriding SensorModelLTV::C
	 */
	Mat_zx C(const Vec_x & = Vec_x::Zero()) const override { return Mat_zx::Identity(); }

	/** Get the measurement covariance matrix.
	 * @param x State (not used)
	 * @return Mat_zz
	 * @note Overriding SensorModelLTV::R
	 */
	Mat_zz R(const Vec_x & = Vec_x::Zero()) const override { return R_; }

private:
	const Mat_zz R_; // Measurement covariance matrix
};

} // namespace models
} // namespace vortex