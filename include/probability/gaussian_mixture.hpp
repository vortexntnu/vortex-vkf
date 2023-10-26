/**
 * @file multi_var_gauss.hpp
 * @author Eirik Kolås
 * @brief A class for representing a multivariate Gaussian mixture distribution. 
 * Based on "Fundamentals of SensorFusion" by Edmund Brekke
 * @version 0.1
 * @date 2023-10-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include "multi_var_gauss.hpp"

namespace vortex {
namespace prob {

/** 
 * A class for representing a multivariate Gaussian mixture distribution
 * @tparam n_dim dimentions of the Gaussian
 */
template <int n_dim>
class GaussianMixture {
public:
    using Vector = Eigen::Vector<double, n_dim>;
    using Matrix = Eigen::Matrix<double, n_dim, n_dim>;
    
    GaussianMixture(std::vector<int> weights, std::vector<MultiVarGauss<n_dim>> gaussians)
        : weights_(weights), gaussians_(gaussians) 
    {
        assert(weights_.size() == gaussians_.size());
    }

    /** Calculate the probability density function of x given the Gaussian mixture
     * @param x
     * @return double
     */
    double pdf(const Vector& x) const {
        double pdf = 0;
        for (int i = 0; i < gaussians_.size(); i++) {
            pdf += weights_[i] * gaussians_[i].pdf(x);
        }
        return pdf;
    }

    /** Find the mean of the Gaussian mixture
     * @return Vector 
    */
    Vector mean() const { 
        Vector mean = Vector::Zero();
        for (int i = 0; i < gaussians_.size(); i++) {
            mean += weights_[i] * gaussians_[i].mean();
        }
        return mean;
    }

    /** Find the covariance of the Gaussian mixture
     * @return Matrix 
    */
    Matrix cov() const { 
        // Spread of innovations
        Matrix P_bar = Matrix::Zero();
        for (int i = 0; i < gaussians_.size(); i++) {
            P_bar += weights_[i] * gaussians_[i].mean() * gaussians_[i].mean().transpose();
        }
        P_bar -= mean() * mean().transpose();

        // Spread of Gaussians
        Matrix P = Matrix::Zero();
        for (int i = 0; i < gaussians_.size(); i++) {
            P += weights_[i] * gaussians_[i].cov();
        }
        return P + P_bar;
    }

    /** Reduce the Gaussian mixture to a single Gaussian
     * @return MultiVarGauss
    */
    MultiVarGauss<n_dim> reduce() const {
        return MultivarGauss(mean(), cov());
    }

    /** dimentions of the Gaussian mixture
     * @return int 
    */
    int n_dims() const { return (n_dim); }

private: 
    std::vector<int> weights_;
    std::vector<MultiVarGauss<n_dim>> gaussians_;
    

};  // class GaussianMixture
}  // namespace probability
}  // namespace vortex