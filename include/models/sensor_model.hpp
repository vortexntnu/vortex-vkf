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

#include <Eigen/Dense>
#include <probability/multi_var_gauss.hpp>

namespace vortex {
namespace filters {


template <int n_dim_x, int n_dim_z>
class SensorModel {
    using Measurement = Eigen::Matrix<double, n_dim_z, 1>;
    using State = Eigen::Matrix<double, n_dim_x, 1>;
    using Mat_zz = Eigen::Matrix<double, n_dim_z, n_dim_z>;
    using Mat_zx = Eigen::Matrix<double, n_dim_z, n_dim_x>;
    using Mat_xx = Eigen::Matrix<double, n_dim_x, n_dim_x>;
public:

    virtual ~SensorModel() = default;

    virtual Measurement h(const State& x) const = 0;
    vitrual Mat_zx H(const State& x) const = 0;
    virtual Mat_zz R(const State& x) const = 0;

    /**
     * @brief Get the predicted measurement distribution given a state estimate
     * 
     * @param x_est State estimate
     * @return prob::MultiVarGauss 
     */
    virtual prob::MultiVarGauss pred_from_est(const prob::MultiVarGauss<n_dim_x>& x_est) const 
    {
        Mat_xx P = x_est.cov();
        Mat_zx H = this->H(x_est.mean());
        Mat_zz R = this->R(x_est.mean());

        return {this->h(x_est.mean()), H * P * H.transpose() + R};
    }

    /**
     * @brief Get the predicted measurement distribution given a state
     * @param x State
     * @return prob::MultiVarGauss 
     */
    virtual prob::MultiVarGauss pred_from_state(const State& x) const 
    {
        return {this->h(x), this->R(x)};
    }
};

}  // namespace filters
}  // namespace vortex