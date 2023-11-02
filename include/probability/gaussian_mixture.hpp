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
#include <probability/multi_var_gauss.hpp>

namespace vortex {
namespace prob {

/** 
 * A class for representing a multivariate Gaussian mixture distribution
 * @tparam N_DIM_x dimentions of the Gaussian
 */
template <int N_DIM_x>
class GaussianMixture {
public:
    using Vec = Eigen::Vector<double, N_DIM_x>;
    using Mat = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    
    GaussianMixture(std::vector<double> weights, std::vector<MultiVarGauss<N_DIM_x>> gaussians)
        : weights_(std::move(weights)), gaussians_(std::move(gaussians)) 
    {
        assert(weights_.size() == gaussians_.size());
    }

    /** Calculate the probability density function of x given the Gaussian mixture
     * @param x
     * @return double
     */
    double pdf(const Vec& x) const {
        double pdf = 0;
        for (int i = 0; i < gaussians_.size(); i++) {
            pdf += weights_[i] * gaussians_[i].pdf(x);
        }
        return pdf;
    }

    /** Find the mean of the Gaussian mixture
     * @return Vector 
    */
    Vec mean() const { 
        Vec mean = Vec::Zero();
        for (int i = 0; i < gaussians_.size(); i++) {
            mean += weights_[i] * gaussians_[i].mean();
        }
        return mean;
    }

    /** Find the covariance of the Gaussian mixture
     * @return Matrix 
    */
    Mat cov() const { 
        // Spread of innovations
        Mat P_bar = Mat::Zero();
        for (int i = 0; i < gaussians_.size(); i++) {
            P_bar += weights_[i] * gaussians_[i].mean() * gaussians_[i].mean().transpose();
        }
        P_bar -= mean() * mean().transpose();

        // Spread of Gaussians
        Mat P = Mat::Zero();
        for (int i = 0; i < gaussians_.size(); i++) {
            P += weights_[i] * gaussians_[i].cov();
        }
        return P + P_bar;
    }

    /** Reduce the Gaussian mixture to a single Gaussian
     * @return MultiVarGauss
    */
    MultiVarGauss<N_DIM_x> reduce() const {
        return MultivarGauss(mean(), cov());
    }

    /** dimentions of the Gaussian mixture
     * @return int 
    */
    int n_dims() const { return (N_DIM_x); }

    /** Get the weights of the Gaussian mixture
     * @return std::vector<int> 
    */
    std::vector<double> weights() const { return weights_; }

    /** Get the Gaussians of the Gaussian mixture
     * @return std::vector<MultiVarGauss<N_DIM_x>> 
    */
    std::vector<MultiVarGauss<N_DIM_x>> gaussians() const { return gaussians_; }

private: 
    const std::vector<double> weights_;
    const std::vector<MultiVarGauss<N_DIM_x>> gaussians_;
    

};  // class GaussianMixture

}  // namespace probability
}  // namespace vortex
