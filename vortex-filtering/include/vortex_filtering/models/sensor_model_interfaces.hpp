/**
 * @file sensor_model.hpp
 * @author Eirik Kolås
 * @brief Sensor model interface. Based on "Fundamentals of Sensor Fusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-10-26
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <memory>
#include <random>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex::models {
namespace interface {

/**
 * @brief Interface for sensor models.
 * @tparam n_dim_x State dimension
 * @tparam n_dim_z Measurement dimension
 * @tparam n_dim_w Measurement noise dimension (Default: n_dim_z)
 * @note To derive from this class, you must override the following functions:
 * @note - h
 * @note - R
 */
template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_w = n_dim_z> class SensorModel {
public:
  using SensModI = SensorModel<n_dim_x, n_dim_z, n_dim_w>;

  static constexpr int N_DIM_x = (int)n_dim_x; // Declare so that children of this class can reference it
  static constexpr int N_DIM_z = (int)n_dim_z; // Declare so that children of this class can reference it
  static constexpr int N_DIM_w = (int)n_dim_w; // Declare so that children of this class can reference it

  using Vec_x = Eigen::Vector<double, N_DIM_x>;
  using Vec_z = Eigen::Vector<double, N_DIM_z>;
  using Vec_w = Eigen::Vector<double, N_DIM_w>;

  using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
  using Mat_xz = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
  using Mat_xw = Eigen::Matrix<double, N_DIM_x, N_DIM_w>;

  using Mat_zx = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
  using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
  using Mat_zw = Eigen::Matrix<double, N_DIM_z, N_DIM_w>;

  using Mat_ww = Eigen::Matrix<double, N_DIM_w, N_DIM_w>;

  using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
  using Gauss_z = prob::MultiVarGauss<N_DIM_z>;
  using Gauss_w = prob::MultiVarGauss<N_DIM_w>;

  using SharedPtr = std::shared_ptr<SensModI>;

  SensorModel()          = default;
  virtual ~SensorModel() = default;

  /**
   * @brief Sensor Model
   * @param x State
   * @return Vec_z
   */
  virtual Vec_z h(const Vec_x &x, const Vec_w &w) const = 0;

  /**
   * @brief Noise covariance matrix
   * @param x State
   * @return Mat_zz
   */
  virtual Mat_ww R(const Vec_x &x) const = 0;

  /** Sample from the sensor model
   * @param x State
   * @param w Noise
   * @param gen Random number generator (For deterministic behaviour)
   * @return Vec_z
   */
  Vec_z sample_h(const Vec_x &x, std::mt19937 &gen) const
  {
    prob::MultiVarGauss<N_DIM_w> w = {Vec_w::Zero(), R(x)};
    return this->h(x, w.sample(gen));
  }
};

/**
 * @brief Linear Time Varying Sensor Model Interface. [z = Cx + Hw]
 * @tparam n_dim_x State dimension
 * @tparam n_dim_z Measurement dimension
 * @tparam n_dim_w Measurement noise dimension (Default: n_dim_z)
 * @note To derive from this class, you must override the following functions:
 * @note - h (optional)
 * @note - C
 * @note - R
 * @note - H (optional if N_DIM_x == N_DIM_z)
 */
template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_w = n_dim_z> class SensorModelLTV : public SensorModel<n_dim_x, n_dim_z, n_dim_z> {
public:
  using SensModI               = SensorModel<n_dim_x, n_dim_z, n_dim_w>;
  static constexpr int N_DIM_x = SensModI::N_DIM_x;
  static constexpr int N_DIM_z = SensModI::N_DIM_z;
  static constexpr int N_DIM_w = SensModI::N_DIM_w;

  using Vec_z = typename SensModI::Vec_z;
  using Vec_x = typename SensModI::Vec_x;
  using Vec_w = typename SensModI::Vec_w;

  using Mat_xx = typename SensModI::Mat_xx;
  using Mat_zx = typename SensModI::Mat_zx;
  using Mat_zz = typename SensModI::Mat_zz;
  using Mat_zw = typename SensModI::Mat_zw;

  using Mat_ww = typename SensModI::Mat_ww;

  using Gauss_x = typename SensModI::Gauss_x;
  using Gauss_z = typename SensModI::Gauss_z;

  using SharedPtr = std::shared_ptr<SensorModelLTV>;

  virtual ~SensorModelLTV() = default;
  /** Sensor Model
   * Overriding SensorModel::h
   * @param x State
   * @param w Noise
   * @return Vec_z
   */
  virtual Vec_z h(const Vec_x &x, const Vec_w &w = Vec_w::Zero()) const override
  {
    Mat_zx C = this->C(x);
    Mat_zw H = this->H(x);
    return C * x + H * w;
  }

  /**
   * @brief Jacobian of sensor model with respect to state
   * @param x State
   * @return Mat_zx
   */
  virtual Mat_zx C(const Vec_x &x) const = 0;

  /**
   * @brief Noise matrix
   * @param x State
   * @return Mat_zz
   */
  virtual Mat_zw H(const Vec_x &x = Vec_x::Zero()) const
  {
    if (N_DIM_x != N_DIM_z) {
      throw std::runtime_error("SensorModelLTV::H not implemented");
    }
    (void)x; // unused
    return Mat_zw::Identity();
  }

  /**
   * @brief Noise covariance matrix
   * @param x State
   * @return Mat_zz
   */
  virtual Mat_ww R(const Vec_x &x) const override = 0;

  /**
   * @brief Get the predicted measurement distribution given a state estimate. Updates the covariance
   *
   * @param x_est Vec_x estimate
   * @return prob::MultiVarGauss
   */
  Gauss_z pred_from_est(const Gauss_x &x_est) const
  {
    Mat_xx P = x_est.cov();
    Mat_zx C = this->C(x_est.mean());
    Mat_ww R = this->R(x_est.mean());
    Mat_zw H = this->H(x_est.mean());

    return {this->h(x_est.mean()), C * P * C.transpose() + H * R * H.transpose()};
  }

  /**
   * @brief Get the predicted measurement distribution given a state. Does not update the covariance
   * @param x Vec_x
   * @return prob::MultiVarGauss
   */
  Gauss_z pred_from_state(const Vec_x &x) const
  {
    Mat_ww R = this->R(x);
    Mat_zw H = this->H(x);
    return {this->h(x), H * R * H.transpose()};
  }
};

} // namespace interface

namespace concepts {

template <typename T>
concept SensorModel = requires {
  // Has member type SensModI
  typename T::SensModI;
  // Derived from SensorModel
  std::derived_from<T, interface::SensorModel<T::SensModI::N_DIM_x, T::SensModI::N_DIM_z, T::SensModI::N_DIM_w>>;
};

template <typename T>
concept SensorModelLTV = requires {
  // Has member type SensModI
  typename T::SensModI;
  // Derived from SensorModelLTV
  std::derived_from<T, interface::SensorModelLTV<T::SensModI::N_DIM_x, T::SensModI::N_DIM_z, T::SensModI::N_DIM_w>>;
};

} // namespace concepts
} // namespace vortex::models