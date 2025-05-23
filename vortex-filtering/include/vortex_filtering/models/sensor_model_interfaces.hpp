/**
 * @file sensor_model.hpp
 * @author Eirik Kolås
 * @brief Sensor model interface. Based on "Fundamentals of Sensor Fusion" by
 * Edmund Brekke
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
#include <vortex_filtering/models/state.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

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
template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_w = n_dim_z>
class SensorModel {
   public:
    static constexpr int N_DIM_x = (int)n_dim_x;
    static constexpr int N_DIM_z = (int)n_dim_z;
    static constexpr int N_DIM_w = (int)n_dim_w;

    using T = vortex::Types_xzw<N_DIM_x, N_DIM_z, N_DIM_w>;

    SensorModel() = default;
    virtual ~SensorModel() = default;

    /**
     * @brief Sensor Model
     * @param x State
     * @return Vec_z
     */
    virtual T::Vec_z h(const T::Vec_x& x,
                       const T::Vec_w& w = T::Vec_w::Zero()) const = 0;

    /**
     * @brief Noise covariance matrix
     * @param x State
     * @return Mat_zz
     */
    virtual T::Mat_ww R(const T::Vec_x& x) const = 0;

    /** Sample from the sensor model
     * @param x State
     * @param w Noise
     * @param gen Random number generator (For deterministic behaviour)
     * @return Vec_z
     */
    T::Vec_z sample_h(const T::Vec_x& x, std::mt19937& gen) const {
        typename T::Gauss_w w{T::Vec_w::Zero(), R(x)};
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
template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_w = n_dim_z>
class SensorModelLTV : public SensorModel<n_dim_x, n_dim_z, n_dim_w> {
   public:
    static constexpr int N_DIM_x = n_dim_x;
    static constexpr int N_DIM_z = n_dim_z;
    static constexpr int N_DIM_w = n_dim_w;

    using T = vortex::Types_xzw<N_DIM_x, N_DIM_z, N_DIM_w>;

    virtual ~SensorModelLTV() = default;
    /** Sensor Model
     * Overriding SensorModel::h
     * @param x State
     * @param w Noise
     * @return Vec_z
     */
    virtual T::Vec_z h(const T::Vec_x& x,
                       const T::Vec_w& w = T::Vec_w::Zero()) const override {
        typename T::Mat_zx C = this->C(x);
        typename T::Mat_zw H = this->H(x);
        return C * x + H * w;
    }

    /**
     * @brief Jacobian of sensor model with respect to state
     * @param x State
     * @return Mat_zx
     */
    virtual T::Mat_zx C(const T::Vec_x& x) const = 0;

    /**
     * @brief Noise matrix
     * @param x State
     * @return Mat_zz
     */
    virtual T::Mat_zw H(const T::Vec_x& /* x */ = T::Vec_x::Zero()) const {
        if (N_DIM_x != N_DIM_z) {
            throw std::runtime_error("SensorModelLTV::H not implemented");
        }
        return T::Mat_zw::Identity();
    }

    /**
     * @brief Noise covariance matrix
     * @param x State
     * @return Mat_zz
     */
    virtual T::Mat_ww R(const T::Vec_x& x) const override = 0;

    /**
     * @brief Get the predicted measurement distribution given a state estimate.
     * Updates the covariance
     *
     * @param x_est TVec_x estimate
     * @return prob::MultiVarGauss
     */
    T::Gauss_z pred_from_est(const auto& x_est) const
        requires(vortex::concepts::MultiVarGaussLike<decltype(x_est), N_DIM_x>)
    {
        typename T::Mat_xx P = x_est.cov();
        typename T::Mat_zx C = this->C(x_est.mean());
        typename T::Mat_ww R = this->R(x_est.mean());
        typename T::Mat_zw H = this->H(x_est.mean());

        return {this->h(x_est.mean()),
                C * P * C.transpose() + H * R * H.transpose()};
    }

    /**
     * @brief Get the predicted measurement distribution given a state. Does not
     * update the covariance
     * @param x Vec_x
     * @return prob::MultiVarGauss
     */
    T::Gauss_z pred_from_state(const T::Vec_x& x) const {
        typename T::Mat_ww R = this->R(x);
        typename T::Mat_zw H = this->H(x);
        return {this->h(x), H * R * H.transpose()};
    }
};

}  // namespace interface
}  // namespace vortex::models
