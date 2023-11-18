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
#include <random>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace models {

class SensorModelX {
public:
    // Using dynamic Eigen types
    using VecX = Eigen::VectorXd;
    using MatXX = Eigen::MatrixXd;
    using GaussX = prob::MultiVarGauss<Eigen::Dynamic>;

    // Constructor to initialize the dimensions
    SensorModelX(int dim_x, int dim_z, int dim_w)
        : dim_x_(dim_x), dim_z_(dim_z), dim_w_(dim_w) {}

    virtual ~SensorModelX() = default;

    /**
     * @brief Sensor Model
     * @param x State
     * @param w Noise
     * @return Vec_z
     */
    virtual VecX hX(const VecX& x, const VecX& w) const = 0;

    /**
     * @brief Noise covariance matrix. (pure virtual function)
     * @param x State
     * @return Mat_zz R
     */
    virtual MatXX RX(const VecX& x) const = 0;

    /** Sample from the sensor model
     * @param x State
     * @param w Noise
     * @param gen Random number generator (For deterministic behaviour)
     * @return Vec_z
     */
    VecX sample_hX(const VecX& x, std::mt19937& gen) const {
        GaussX w = {VecX::Zero(dim_w_), RX(x)};
        return hX(x, w.sample(gen));
    }

    /** Sample from the sensor model
     * @param x State
     * @return Vec_z
     */
    VecX sample_hX(const VecX& x) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        return sample_hX(x, gen);
    }

    int get_dim_x() const { return dim_x_; }
    int get_dim_z() const { return dim_z_; }
    int get_dim_w() const { return dim_w_; }

protected:
    const int dim_x_;  // State dimension
    const int dim_z_;  // Measurement dimension
    const int dim_w_;  // Process noise dimension
};

template <int n_dim_x, int n_dim_z, int n_dim_w>
/**
 * @brief Interface for sensor models.
 * 
 */
class SensorModelI : public SensorModelX {
public:
    static constexpr int N_DIM_x = n_dim_x; // Declare so that children of this class can reference it
    using Vec_x  = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;

    static constexpr int N_DIM_z = n_dim_z; // Declare so that children of this class can reference it
    using Vec_z  = Eigen::Vector<double, N_DIM_z>;
    using Mat_zx = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_xz = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
    using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;

    static constexpr int N_DIM_w = n_dim_w; // Declare so that children of this class can reference it
    using Vec_w  = Eigen::Vector<double, N_DIM_w>;
    using Mat_xw = Eigen::Matrix<double, N_DIM_x, N_DIM_w>;
    using Mat_zw = Eigen::Matrix<double, N_DIM_z, N_DIM_w>;
    using Mat_ww = Eigen::Matrix<double, N_DIM_w, N_DIM_w>;
    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_z = prob::MultiVarGauss<N_DIM_z>;
    using Gauss_w = prob::MultiVarGauss<N_DIM_w>;

    SensorModelI() : SensorModelX(N_DIM_x, N_DIM_z, N_DIM_w) {}
    virtual ~SensorModelI() = default;

    /**
     * @brief Sensor Model
     * @param x State
     * @return Vec_z
     */
    virtual Vec_z h(const Vec_x& x, const Vec_w& w) const = 0;

    /**
     * @brief Noise covariance matrix
     * @param x State
     * @return Mat_zz
     */
    virtual Mat_ww R(const Vec_x& x) const = 0;

    /** Sample from the sensor model
     * @param x State
     * @param w Noise
     * @param gen Random number generator (For deterministic behaviour)
     * @return Vec_z
     */
    Vec_z sample_h(const Vec_x& x, std::mt19937& gen) const
    {
        prob::MultiVarGauss<N_DIM_w> w = {Vec_w::Zero(), R(x)};
        return this->h(x, w.sample(gen));
    }

    /** Sample from the sensor model
     * @param x State
     * @return Vec_z
     */
    Vec_z sample_h(const Vec_x& x) const
    {
        std::random_device rd;                            
        std::mt19937 gen(rd());                             
        return sample_h(x, gen);
    }

    // Override dynamic size functions to use static size functions
protected:
    // Discrete time dynamics (pure virtual function)
    virtual VecX hX(const VecX& x, const VecX& w) const override
    {
        return h(x, w);
    }

    // Discrete time process noise (pure virtual function)
    virtual MatXX RX(const VecX& x) const override
    {
        return R(x);
    }

};



template <int n_dim_x, int n_dim_z>
/**
 * @brief Interface for sensor models.
 * 
 */
class SensorModelEKFI : public SensorModelI<n_dim_x, n_dim_z, n_dim_z>{
public:
    static constexpr int N_DIM_x = n_dim_x; // Declare so that children of this class can reference it
    static constexpr int N_DIM_z = n_dim_z; // Declare so that children of this class can reference it
    static constexpr int N_DIM_w = n_dim_z; // Declare so that children of this class can reference it
    using Vec_z  = Eigen::Vector<double, N_DIM_z>;
    using Vec_x  = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Mat_zx = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
    using Mat_xz = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
    using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
    using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
    using Gauss_z = prob::MultiVarGauss<N_DIM_z>;

    virtual ~SensorModelEKFI() = default;
    /** Sensor Model
     * Overriding SensorModelI::h
     * @param x State
     * @param w Noise
     * @return Vec_z
     */
    Vec_z h(const Vec_x& x, const Vec_z& w) const override
    {
        return this->h(x) + w;
    }
    
    /**
     * @brief Sensor Model
     * @param x State
     * @return Vec_z
     */
    virtual Vec_z h(const Vec_x& x) const = 0;
    /**
     * @brief Jacobian of sensor model with respect to state
     * @param x State
     * @return Mat_zx 
     */
    virtual Mat_zx H(const Vec_x& x) const = 0;
    /**
     * @brief Noise covariance matrix
     * @param x State
     * @return Mat_zz 
     */
    virtual Mat_zz R(const Vec_x& x) const override = 0;

    /**
     * @brief Get the predicted measurement distribution given a state estimate. Updates the covariance
     * 
     * @param x_est Vec_x estimate
     * @return prob::MultiVarGauss 
     */
    virtual Gauss_z pred_from_est(const Gauss_x& x_est) const 
    {
        Mat_xx P = x_est.cov();
        Mat_zx H = this->H(x_est.mean());
        Mat_zz R = this->R(x_est.mean());

        return {this->h(x_est.mean()), H * P * H.transpose() + R};
    }

    /**
     * @brief Get the predicted measurement distribution given a state. Does not update the covariance
     * @param x Vec_x
     * @return prob::MultiVarGauss 
     */
    virtual Gauss_z pred_from_state(const Vec_x& x) const 
    {
        return {this->h(x), this->R(x)};
    }

};


}  // namespace models
}  // namespace vortex