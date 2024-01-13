#pragma once
#include <Eigen/Dense>
#include <random>

namespace vortex {
namespace prob {

/**
 * A class for representing a multivariate Gaussian distribution
 * @tparam N_DIMS dimentions of the Gaussian
 */
template <int n_dims> class MultiVarGauss {
public:
  using Vec_n  = Eigen::Vector<double, n_dims>;
  using Mat_nn = Eigen::Matrix<double, n_dims, n_dims>;

  /** Construct a Gaussian with a given mean and covariance matrix
   * @param mean
   * @param cov Symmetric positive definite covariance matrix
   */
  MultiVarGauss(const Vec_n &mean, const Mat_nn &cov)
      : N_DIMS(mean.size()), mean_(mean), cov_(cov), cov_inv_(cov_.llt().solve(Mat_nn::Identity(size(), size())))
  {
    // Check that the covariance matrix is positive definite and symmetric
    if (!cov_.isApprox(cov_.transpose(), 1e-6)) {
      throw std::invalid_argument("Covariance matrix is not symmetric");
    }
    if (cov_.llt().info() != Eigen::Success) {
      throw std::invalid_argument("Covariance matrix is not positive definite");
    }
  }

  // Default constructor
  MultiVarGauss() = default;

  // Copy constructor
  MultiVarGauss(const MultiVarGauss &other) : N_DIMS(other.N_DIMS), mean_(other.mean_), cov_(other.cov_), cov_inv_(other.cov_inv_) {}

  // Conversion constructor to convert dynamic size Gaussians to static size Gaussians
  template <int N> MultiVarGauss(const MultiVarGauss<N> &other)
  {
    if (n_dims != Eigen::Dynamic) {
      if (n_dims != other.size()) {
        throw std::invalid_argument("Cannot convert Gaussians of different sizes");
      }
    }

    N_DIMS = other.size();

    Vec_n mean = other.mean();
    Mat_nn cov = other.cov();

    // cov_inv_ = other.cov_inv();

    *this = {mean, cov};
  }

  // Conversion operator to convert static size Gaussians to dynamic size Gaussians
  operator MultiVarGauss<Eigen::Dynamic>() const { return {this->mean_, this->cov_}; }

  // Copy assignment operator
  MultiVarGauss &operator=(const MultiVarGauss &other)
  {
    if (&other != this) {
      // Copy the data from 'other' to 'this'
      this->mean_    = other.mean_;
      this->cov_     = other.cov_;
      this->cov_inv_ = other.cov_inv_;
    }
    return *this;
  }

  /** Calculate the probability density function of x given the Gaussian
   * @param x
   * @return double
   */
  double pdf(const Vec_n &x) const
  {
    const Vec_n diff      = x - mean();
    const double exponent = -0.5 * diff.transpose() * cov_inv() * diff;
    return std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, size()) * cov().determinant());
  }

  /** Calculate the log likelihood of x given the Gaussian.
   * Assumes that the covariance matrix is positive definite and symmetric
   * @param x
   * @return double
   */
  double logpdf(const Vec_n &x) const
  {
    const Vec_n diff      = x - mean();
    const double exponent = -0.5 * diff.transpose() * cov_inv() * diff;
    return exponent - 0.5 * std::log(std::pow(2 * M_PI, size()) * cov().determinant());
  }

  Vec_n mean() const { return mean_; }
  Mat_nn cov() const { return cov_; }
  Mat_nn cov_inv() const { return cov_inv_; }

  /** Calculate the Mahalanobis distance of x given the Gaussian
   * @param x
   * @return double
   */
  double mahalanobis_distance(const Vec_n &x) const
  {
    const Vec_n diff = x - mean();
    return std::sqrt(diff.transpose() * cov_inv() * diff);
  }

  /** Sample from the Gaussian
   * @param gen Random number generator
   * @return Vector
   */
  Vec_n sample(std::mt19937 &gen) const
  {
    std::normal_distribution<> d{0, 1};
    Vec_n sample(size());
    for (int i = 0; i < size(); ++i) {
      sample(i) = d(gen);
    }
    return mean() + cov().llt().matrixL() * sample;
  }

  /** Sample from the Gaussian
   * @return Vector
   */
  Vec_n sample() const
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    return sample(gen);
  }

  /** size of the Gaussian
   * @return int
   */
  int size() const { return N_DIMS; }

  /** Construct a Standard Gaussian Distribution with zero mean and identity covariance matrix
   * @return MultiVarGauss
   */
  static MultiVarGauss<n_dims> Standard() { return MultiVarGauss<n_dims>(Vec_n::Zero(), Mat_nn::Identity()); }

private:
  size_t N_DIMS;
  Vec_n mean_;
  Mat_nn cov_;
  Mat_nn cov_inv_;
};

using MultiVarGaussXd = MultiVarGauss<Eigen::Dynamic>;
using MultiVarGauss2d = MultiVarGauss<2>;
using MultiVarGauss3d = MultiVarGauss<3>;
using MultiVarGauss4d = MultiVarGauss<4>;

using GaussXd = MultiVarGaussXd;
using Gauss2d = MultiVarGauss2d;
using Gauss3d = MultiVarGauss3d;
using Gauss4d = MultiVarGauss4d;

} // namespace prob
} // namespace vortex
