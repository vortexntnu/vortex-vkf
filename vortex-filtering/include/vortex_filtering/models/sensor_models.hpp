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
template<int n_dim_x, int n_dim_z>
class IdentitySensorModel : public vortex::models::SensorModelEKFI<n_dim_x, n_dim_z> {
public:
    using SensModI = vortex::models::SensorModelEKFI<n_dim_x, n_dim_z>;

    using typename SensModI::Vec_z;
    using typename SensModI::Vec_x;
    using typename SensModI::Mat_xx;
    using typename SensModI::Mat_zx;
    using typename SensModI::Mat_zz;
    using SensModI::N_DIM_x;
    using SensModI::N_DIM_z;

    /** Construct a new Simple Sensor Model object. 
     * The measurement model is simply the n_dim_z first elements of the state vector.
     * @param std Standard deviation. Sets the measurement covariance matrix R to I*std^2.
     * @tparam n_dim_x Dimension of state
     * @tparam n_dim_z Dimension of measurement
     */
    IdentitySensorModel(double std) : R_(Mat_zz::Identity()*std*std) {}

    /** Construct a new Simple Sensor Model object.
     * The measurement model is simply the n_dim_z first elements of the state vector.
     * @param R Measurement covariance matrix
     */
    IdentitySensorModel(Mat_zz R) : R_(R) {}

    /** Get the predicted measurement given a state estimate.
     * Overriding SensorModelEKFI::h
     * @param x State
     * @return Vec_z 
     */
    Vec_z h(const Vec_x& x) const override { return H()*x; }

    /** Get the Jacobian of the measurement model with respect to the state.
     * @param x State
     * @return Mat_zx 
     */
    Mat_zx H() const { return Mat_zx::Identity(); }

    /** Get the measurement covariance matrix.
     * @return Mat_zz 
     */
    Mat_zz R() const { return R_; }



private:
    /** Get the Jacobian of the measurement model with respect to the state
     * Overriding SensorModelEKFI::H
     * @param x State
     * @return Mat_zx 
     */
    Mat_zx H(const Vec_x&) const override { return H(); }

    /** Get the measurement covariance matrix
     * Overriding SensorModelEKFI::R 
     * @return Mat_zz 
     */
    Mat_zz R(const Vec_x&) const override { return R(); }

    const Mat_zz R_; // Measurement covariance matrix
};

} // namespace models
} // namespace vortex