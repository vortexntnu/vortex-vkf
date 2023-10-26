#pragma once
#include <eigen3/Eigen/Dense>

namespace vortex {
namespace probability {

/** 
 * A class for representing a multivariate Gaussian distribution
 * @tparam n_dim dimentions of the Gaussian
 */
template <int n_dim>
class MultiVarGauss {
public:
    using Vector = Eigen::Vector<double, n_dim>;
    using Matrix = Eigen::Matrix<double, n_dim, n_dim>;
    
    MultiVarGauss(const Vector& mean, const Matrix& cov)
        : mean_(mean), cov_(cov) 
    {
        // Check that the covariance matrix is positive definite and symmetric
        if (cov_ != cov_.transpose()) {
            throw std::invalid_argument("Covariance matrix is not symmetric");
        }
        if (cov_.llt().info() != Eigen::Success) {
            throw std::invalid_argument("Covariance matrix is not positive definite");
        }
    }
    
    /** Calculate the probability density function of x given the Gaussian
     * @param x
     * @return double
     */
    double pdf(const Vector& x) const {
        const auto diff = x - mean_;
        const auto cov_inv = cov_.llt().solve(Matrix::Identity());
        const auto exponent = -0.5 * diff.transpose() * cov_inv * diff;
        return std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, n_dim) * cov_.determinant());
    }

    /** Calculate the log likelihood of x given the Gaussian.
     * Assumes that the covariance matrix is positive definite and symmetric
     * @param x 
     * @return double 
     */
    double logpdf(const Vector& x) const {
        const auto diff = x - mean_;
        const auto cov_inv = cov_.llt().solve(Matrix::Identity());
        const auto exponent = -0.5 * diff.transpose() * cov_inv * diff;
        return exponent - 0.5 * std::log(std::pow(2 * M_PI, n_dim) * cov_.determinant());
    }
    

    Vector mean() const { return mean_; }
    Matrix cov() const { return cov_; }

    /** Calculate the Mahalanobis distance of x given the Gaussian
     * @param x 
     * @return double 
     */
    double mahalanobis_distance(const Vector& x) const {
        const auto diff = x - mean_;
        const auto cov_inv = cov_.llt().solve(Matrix::Identity());
        return std::sqrt(diff.transpose() * cov_inv * diff);
    }


    /** dimentions of the Gaussian
     * @return int 
    */
    int n_dims() const { return n_dim; }
    
    private:
    Vector mean_;
    Matrix cov_;
};

}  // namespace probability
} // namespace vortex
