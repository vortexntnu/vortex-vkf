#pragma once
#include <Eigen/Dense>
#include <random>

namespace vortex {
namespace prob {

/** 
 * A class for representing a multivariate Gaussian distribution
 * @tparam N_DIMS dimentions of the Gaussian
 */
template <int n_dims_>
class MultiVarGauss {
public:
    using Vector = Eigen::Vector<double, n_dims_>;
    using Matrix = Eigen::Matrix<double, n_dims_, n_dims_>;
    
    MultiVarGauss(const Vector& mean = Vector::Zero(), const Matrix& cov = Matrix::Identity())
        : mean_(mean), cov_(cov), actual_n_dims_(cov_.rows()), cov_inv_(cov_.llt().solve(Matrix::Identity(n_dims(), n_dims())))
    {
        // Check that the covariance matrix is positive definite and symmetric
        if (!cov_.isApprox(cov_.transpose(), 1e-6)) {
            throw std::invalid_argument("Covariance matrix is not symmetric");
        }
        if (cov_.llt().info() != Eigen::Success) {
            throw std::invalid_argument("Covariance matrix is not positive definite");
        }
    }

    // Conversion constructor to convert dynamic size Gaussians to static size Gaussians
    MultiVarGauss(const MultiVarGauss<-1>& other) {
        assert(other.mean().size() == n_dims_);
        assert(other.cov().rows() == n_dims_ && other.cov().cols() == n_dims_);

        // Assign or transform other's data to this instance
        this->mean_ = other.mean();
        this->cov_ = other.cov();
    }

    // Conversion operator to convert static size Gaussians to dynamic size Gaussians
    operator MultiVarGauss<-1>() const {   
        return {this->mean_, this->cov_};
    }
    
    /** Calculate the probability density function of x given the Gaussian
     * @param x
     * @return double
     */
    double pdf(const Vector& x) const {
        const auto diff = x - mean();
        const auto exponent = -0.5 * diff.transpose() * cov_inv() * diff;
        return std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, n_dims()) * cov().determinant());
    }

    /** Calculate the log likelihood of x given the Gaussian.
     * Assumes that the covariance matrix is positive definite and symmetric
     * @param x 
     * @return double 
     */
    double logpdf(const Vector& x) const {
        const auto diff = x - mean();
        const auto exponent = -0.5 * diff.transpose() * cov_inv() * diff;
        return exponent - 0.5 * std::log(std::pow(2 * M_PI, n_dims()) * cov().determinant());
    }
    

    Vector mean() const { return mean_; }
    Matrix cov() const { return cov_; }
    Matrix cov_inv() const { return cov_inv_; }

    /** Calculate the Mahalanobis distance of x given the Gaussian
     * @param x 
     * @return double 
     */
    double mahalanobis_distance(const Vector& x) const {
        const auto diff = x - mean();
        return std::sqrt(diff.transpose() * cov_inv() * diff);
    }

    /** Sample from the Gaussian
     * @param gen Random number generator
     * @return Vector 
     */
    Vector sample(std::mt19937& gen) const {
        std::normal_distribution<> d{0, 1};
        Vector sample;
        for (int i = 0; i < n_dims(); ++i) {
            sample(i) = d(gen);
        }
        return mean() + cov().llt().matrixL() * sample;
    }

    /** Sample from the Gaussian
     * @return Vector 
     */
    Vector sample() const {
        std::random_device rd;                            
        std::mt19937 gen(rd());                           
        return sample(gen);
    }


    /** dimentions of the Gaussian
     * @return int 
    */
    int n_dims() const { return actual_n_dims_; }
    
private:
    Vector mean_;
    Matrix cov_;
    size_t actual_n_dims_;
    Matrix cov_inv_;

};

}  // namespace probability
} // namespace vortex
