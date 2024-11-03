#pragma once
#include <Eigen/Dense>
#include <manif/manif.h>
#include <numbers>
#include <random>

namespace vortex::prob {

/**
 * A class for representing a multivariate Gaussian distribution on a Lie group
 * @tparam Derived The specific Lie group type from manif (e.g., SE3d, SO3d)
 */
template <typename Derived> class LieGroupGauss {
public:
  using LieGroup = Derived;
  using Tangent  = typename LieGroup::Tangent;
  using Vec_n    = Eigen::Vector<double, Tangent::DoF>;
  using Mat_nn   = Eigen::Matrix<double, Tangent::DoF, Tangent::DoF>;

  /** Construct a Gaussian on a Lie group with a given mean and covariance matrix in tangent space
   * @param mean Lie group element (e.g., SE3 or SO3 element)
   * @param cov Symmetric positive definite covariance matrix in tangent space
   */
  LieGroupGauss(LieGroup mean, const Mat_nn &cov)
      : mean_(mean)
      , cov_(cov)
      , cov_inv_(cov.llt().solve(Mat_nn::Identity()))
  {
    if (!cov_.isApprox(cov_.transpose(), 1e-6)) {
      throw std::invalid_argument("Covariance matrix is not symmetric");
    }
    if (cov_.llt().info() != Eigen::Success) {
      throw std::invalid_argument("Covariance matrix is not positive definite");
    }
  }

  // Default constructor
  LieGroupGauss()
      : mean_(LieGroup::Identity())
      , cov_(Mat_nn::Identity())
      , cov_inv_(cov_.llt().solve(Mat_nn::Identity()))
  {
  }

  /** Calculate the probability density of x given the Gaussian
   * @param x Lie group element
   * @return double
   */
  double pdf(const LieGroup &x) const
  {
    Tangent diff    = x - mean_;
    double exponent = -0.5 * diff.coeffs().transpose() * cov_inv_ * diff.coeffs();
    return std::exp(exponent) / std::sqrt(std::pow(2 * std::numbers::pi, Tangent::DoF) * cov_.determinant());
  }

  /** Calculate the log likelihood of x given the Gaussian
   * @param x Lie group element
   * @return double
   */
  double logpdf(const LieGroup &x) const
  {
    Tangent diff    = x - mean_;
    double exponent = -0.5 * diff.coeffs().transpose() * cov_inv_ * diff.coeffs();
    return exponent - 0.5 * std::log(std::pow(2 * std::numbers::pi, Tangent::DoF) * cov_.determinant());
  }

  LieGroup mean() const { return mean_; }
  Mat_nn cov() const { return cov_; }
  Mat_nn cov_inv() const { return cov_inv_; }

  /** Calculate the Mahalanobis distance of x given the Gaussian
   * @param x Lie group element
   * @return double
   */
  double mahalanobis_distance(const LieGroup &x) const
  {
    Tangent diff = x - mean_;
    return std::sqrt(diff.coeffs().transpose() * cov_inv_ * diff.coeffs());
  }

  /** Sample from the Gaussian
   * @param gen Random number generator
   * @return LieGroup element
   */
  LieGroup sample(std::mt19937 &gen) const
  {
    std::normal_distribution<> d{0, 1};
    Vec_n sample(Tangent::DoF);
    for (int i = 0; i < Tangent::DoF; ++i) {
      sample(i) = d(gen);
    }
    Tangent tangent_sample = cov_.llt().matrixL() * sample;
    return mean_ + tangent_sample;
  }

  /** Construct a Standard Gaussian Distribution with identity mean and identity covariance matrix
   * @return LieGroupGauss
   */
  static LieGroupGauss<Derived> Standard() { return {LieGroup::Identity(), Mat_nn::Identity()}; }

  /** operator==
   * @param lhs
   * @param rhs
   * @return bool true if the means and covariances are equal
   */
  friend bool operator==(const LieGroupGauss &lhs, const LieGroupGauss &rhs) { return lhs.mean() == rhs.mean() && lhs.cov() == rhs.cov(); }

  /** operator<<
   * @param os
   * @param gauss
   * @return std::ostream&
   */
  friend std::ostream &operator<<(std::ostream &os, const LieGroupGauss &gauss)
  {
    os << "Mean:\n"
       << gauss.mean() << "\n"
       << "Covariance:\n"
       << gauss.cov();
    return os;
  }

private:
  LieGroup mean_;
  Mat_nn cov_;
  Mat_nn cov_inv_;
};

} // namespace vortex::prob
