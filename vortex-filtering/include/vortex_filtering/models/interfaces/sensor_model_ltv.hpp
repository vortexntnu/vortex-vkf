#pragma once
#include "sensor_model.hpp"

namespace vortex::model::interface {

/**
 * @brief Linear Time Varying Sensor Model Interface on Lie Groups.
 *        [z = C * x + H * w]
 * @tparam Type_x Lie group type of the state
 * @tparam Type_z Lie group type of the measurement
 * @tparam Type_w Lie group type of the measurement noise (default to Type_z)
 */
template <typename Type_x, typename Type_z, typename Type_w = Type_z>
class SensorModelLTV : public SensorModel<Type_x, Type_z, Type_w> {
public:
  using Mx = Type_x;
  using Mz = Type_z;
  using Mw = Type_w;

  using Tx = typename Mx::Tangent;
  using Tz = typename Mz::Tangent;
  using Tw = typename Mw::Tangent;

  using T       = vortex::Types_xzw<Mx::DoF, Mz::DoF, Mw::DoF>;
  using Gauss_w = vortex::prob::LieGroupGauss<Mw>;
  using Gauss_z = vortex::prob::LieGroupGauss<Mz>;

  /**
   * @brief Sensor Model with optional noise
   * @param x State (Lie group element)
   * @param w Measurement noise (Lie group element)
   * @return Measurement (Lie group element)
   */
  virtual Mz g(const Mx &x, const Mw &w) const override
  {
    typename Tz b        = this->b();
    typename T::Mat_zw H = this->J_g_w(x);

    Mz z = x.act(b).exp() + (H * w_log).exp();
    return z;
  }

  virtual T::Vec_v b() const = 0;

  /**
   * @brief Jacobian of the sensor model with respect to the state in tangent space
   * @param x State (Lie group element)
   * @return Jacobian matrix in tangent space of measurement
   */
  virtual T::Mat_zx J_g_x(const Mx &x) const = 0;

  /**
   * @brief Jacobian of the sensor model with respect to the noise in tangent space
   * @param x State (Lie group element)
   * @return Jacobian matrix in tangent space of measurement noise
   */
  virtual T::Mat_zw J_g_w(const Mx & /* x */ = Mx::Identity()) const { return T::Mat_zw::Identity(); }

  /**
   * @brief Noise covariance matrix in the tangent space of measurement noise
   * @param x State (Lie group element)
   * @return Covariance matrix in tangent space
   */
  virtual T::Mat_ww R_w(const Mx &x) const override = 0;

  /**
   * @brief Noise covariance matrix in the tangent space of the measurement
   * @param x State (Lie group element)
   * @return Covariance matrix in tangent space
   */
  virtual T::Mat_zz R_z(const Mx &x) const override
  {
    typename T::Mat_zw H = this->J_g_w(x);
    typename T::Mat_ww R = this->R_w(x);
    return H * R * H.transpose();
  }

  /**
   * @brief Get the predicted measurement distribution given a state estimate
   *
   * @param x_est State estimate with mean and covariance
   * @return Predicted measurement distribution
   */
  Gauss_z pred_from_est(const vortex::prob::LieGroupGauss<Mx> &x_est) const
  {
    typename T::Mat_xx P = x_est.cov();
    typename T::Mat_zx C = this->J_g_x(x_est.mean());
    typename T::Mat_ww R = this->R_w(x_est.mean());
    typename T::Mat_zw H = this->J_g_w(x_est.mean());

    return {this->g(x_est.mean()), C * P * C.transpose() + H * R * H.transpose()};
  }

  /**
   * @brief Get the predicted measurement distribution given a state
   * @param x State
   * @return Predicted measurement distribution
   */
  Gauss_z pred_from_state(const Mx &x) const
  {
    typename T::Mat_ww R = this->R_w(x);
    typename T::Mat_zw H = this->J_g_w(x);
    return {this->g(x), H * R * H.transpose()};
  }
};

template <std::size_t DIM_x, std::size_t DIM_z, std::size_t DIM_w>
using SensorModelLTVR = SensorModelLTV<manif::Rn<double, DIM_x>, manif::Rn<double, DIM_z>, manif::Rn<double, DIM_w>>;

} // namespace vortex::model::interface
