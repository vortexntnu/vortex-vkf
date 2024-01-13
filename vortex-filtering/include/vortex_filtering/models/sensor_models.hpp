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
  using typename SensModI::Mat_zw;
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

  /** Get the Jacobian of the measurement model with respect to noise
   * @param x State (not used)
   * @return Mat_zw
   * @note Overriding SensorModelLTV::H
  */
  Mat_zw H(const Vec_x & = Vec_x::Zero()) const override { return Mat_zw::Identity(); }

private:
  const Mat_zz R_; // Measurement covariance matrix
};

/** Range-Bearing sensor model.
 * The measurement model is the range and bearing to the target.
 * The state vector is the position of the target in cartesian coordinates.
 * x = [x_target, y_target]
 * z = [range, bearing]
 */
class RangeBearingSensor : public interface::SensorModelLTV<2, 2> {
public:
  using SensModI = interface::SensorModelI<2, 2>;
  using typename SensModI::Mat_xx;
  using typename SensModI::Mat_zx;
  using typename SensModI::Mat_zz;
  using typename SensModI::Vec_x;
  using typename SensModI::Vec_z;

  /** Range-Bearing sensor model.
   * The measurement model is the range and bearing to the target.
   * The first 2 elements of the state vector are assumed to be the position of the target in cartesian coordinates.
   * x = [x_target, y_target, ...]
   * z = [range, bearing]
   * @tparam n_dim_x Dimension of state
   * @param std_range Standard deviation of range measurement
   * @param std_bearing Standard deviation of bearing measurement
   */
  RangeBearingSensor(double std_range, double std_bearing) : std_range_(std_range), std_bearing_(std_bearing) {}

  /** Get the measurement model.
   * @param x State
   * @param w Noise
   * @return Vec_z
   * @note Overriding SensorModelLTV::h
   */
  Vec_z h(const Vec_x &x, const Vec_w &w = Vec_w::Zero()) const override
  {
    Vec_z z;
    z << std::sqrt(x(0) * x(0) + x(1) * x(1)), std::atan2(x(1), x(0));
    z += w;
    return z;
  }

  /** Get the Jacobian of the measurement model with respect to the state. State is in cartesian coordinates.
   * @param x State
   * @return Mat_zx
   * @note Overriding SensorModelLTV::C
   */
  Mat_zx C(const Vec_x &x) const override
  {
    Mat_zx C;
    // clang-format off
		C << (x(0) / std::sqrt(x(0)*x(0) + x(1)*x(1))), (x(1) / std::sqrt(x(0)*x(0) + x(1)*x(1))),
		     (-x(1) / (x(0)*x(0) + x(1)*x(1)))        , (x(0) / (x(0)*x(0) + x(1)*x(1)));
    // clang-format on
    return C;
  }

  /** Get the measurement covariance matrix.
   * @param x State (not used)
   * @return Mat_zz
   * @note Overriding SensorModelLTV::R
   */
  Mat_zz R(const Vec_x & = Vec_x::Zero()) const override
  {
    Vec_z D;
    D << std_range_ * std_range_, std_bearing_ * std_bearing_;
    Mat_zz R = D.asDiagonal();
    return R;
  }

private:
  double std_range_;
  double std_bearing_;
};

} // namespace models
} // namespace vortex