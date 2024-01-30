#pragma once
#include <Eigen/Dense>
#include <random>

namespace vortex::prob {

/**
 * A class for representing a multivariate Gaussian distribution
 * @tparam N_DIMS dimentions of the Gaussian
 */
template <size_t n_dims> class MultiVarGauss {
public:
  static constexpr int N_DIMS = (int)n_dims;

  using Vec_n  = Eigen::Vector<double, N_DIMS>;
  using Mat_nn = Eigen::Matrix<double, N_DIMS, N_DIMS>;

  /** Construct a Gaussian with a given mean and covariance matrix
   * @param mean
   * @param cov Symmetric positive definite covariance matrix
   */
  MultiVarGauss(const Vec_n &mean, const Mat_nn &cov) : mean_(mean), cov_(cov), cov_inv_(cov_.llt().solve(Mat_nn::Identity()))
  {
    if (!cov_.isApprox(cov_.transpose(), 1e-6)) {
      throw std::invalid_argument("Covariance matrix is not symmetric");
    }
    if (cov_.llt().info() != Eigen::Success) {
      throw std::invalid_argument("Covariance matrix is not positive definite");
    }
  }

  // Default constructor
  MultiVarGauss() = default;

  /** Calculate the probability density function of x given the Gaussian
   * @param x
   * @return double
   */
  double pdf(const Vec_n &x) const
  {
    Vec_n diff      = x - mean();
    double exponent = -0.5 * diff.transpose() * cov_inv() * diff;
    return std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, size()) * cov().determinant());
  }

  /** Calculate the log likelihood of x given the Gaussian.
   * Assumes that the covariance matrix is positive definite and symmetric
   * @param x
   * @return double
   */
  double logpdf(const Vec_n &x) const
  {
    Vec_n diff      = x - mean();
    double exponent = -0.5 * diff.transpose() * cov_inv() * diff;
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
    Vec_n diff = x - mean();
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

  /** Get the number of dimensions of the Gaussian
   * @return int
   */
  static constexpr int size() { return N_DIMS; }

  /** Construct a Standard Gaussian Distribution with zero mean and identity covariance matrix
   * @return MultiVarGauss
   */
  static MultiVarGauss Standard() { return {Vec_n::Zero(), Mat_nn::Identity()}; }

private:
  Vec_n mean_;
  Mat_nn cov_;
  Mat_nn cov_inv_;
};

template <int n_dims> using Gauss = MultiVarGauss<n_dims>;
using Gauss2d                     = Gauss<2>;
using Gauss3d                     = Gauss<3>;
using Gauss4d                     = Gauss<4>;

} // namespace vortex::prob
