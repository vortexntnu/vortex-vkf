#pragma once
#include <Eigen/Dense>

namespace vortex {
namespace prob {

/** 
 * A class for representing a multivariate Gaussian distribution
 * @tparam N_DIMS dimentions of the Gaussian
 */
template <int N_DIMS>
class MultiVarGauss {
public:
    using Vector = Eigen::Vector<double, N_DIMS>;
    using Matrix = Eigen::Matrix<double, N_DIMS, N_DIMS>;
    
    MultiVarGauss(const Vector& mean, const Matrix& cov)
        : mean_(mean), cov_(cov), cov_inv_(cov_.llt().solve(Matrix::Identity(N_DIMS, N_DIMS)))
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
        const auto exponent = -0.5 * diff.transpose() * cov_inv_ * diff;
        return std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, N_DIMS) * cov_.determinant());
    }

    /** Calculate the log likelihood of x given the Gaussian.
     * Assumes that the covariance matrix is positive definite and symmetric
     * @param x 
     * @return double 
     */
    double logpdf(const Vector& x) const {
        const auto diff = x - mean_;
        const auto exponent = -0.5 * diff.transpose() * cov_inv_ * diff;
        return exponent - 0.5 * std::log(std::pow(2 * M_PI, N_DIMS) * cov_.determinant());
    }
    

    Vector mean() const { return mean_; }
    Matrix cov() const { return cov_; }
    Matrix cov_inv() const { return cov_inv_; }

    /** Calculate the Mahalanobis distance of x given the Gaussian
     * @param x 
     * @return double 
     */
    double mahalanobis_distance(const Vector& x) const {
        const auto diff = x - mean_;
        return std::sqrt(diff.transpose() * cov_inv_ * diff);
    }


    /** dimentions of the Gaussian
     * @return int 
    */
    int n_dims() const { return N_DIMS; }
    
    private:
    Vector mean_;
    Matrix cov_;
    Matrix cov_inv_;
};

}  // namespace probability
} // namespace vortex
