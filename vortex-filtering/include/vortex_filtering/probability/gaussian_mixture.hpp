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
#include <numeric> // std::accumulate
#include <vector>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace prob {

/**
 * A class for representing a multivariate Gaussian mixture distribution
 * @tparam n_dims dimentions of the Gaussian
 */
template <int n_dims> class GaussianMixture {
public:
  using Vec_n   = Eigen::Vector<double, n_dims>;
  using Mat_nn  = Eigen::Matrix<double, n_dims, n_dims>;
  using Gauss_n = MultiVarGauss<n_dims>;

  /** Construct a new Gaussian Mixture object
   * @param weights Weights of the Gaussians
   * @param gaussians Gaussians
   * @note The weights are automatically normalized, so they don't need to sum to 1.
   */
  GaussianMixture(Eigen::VectorXd weights, std::vector<Gauss_n> gaussians) : weights_(std::move(weights)), gaussians_(std::move(gaussians)) {}

  /** Construct a new Gaussian Mixture object from a std::vector of weights
   * @param weights Weights of the Gaussians
   * @param gaussians Gaussians
   * @note The weights are automatically normalized, so they don't need to sum to 1.
   */
  GaussianMixture(std::vector<double> weights, std::vector<Gauss_n> gaussians)
      : weights_(Eigen::Map<Eigen::VectorXd>(weights.data(), weights.size())), gaussians_(std::move(gaussians))
  {
  }

  /** Default Constructor
   * weights and gaussians are empty
   */
  GaussianMixture() = default;

  /***/
  /** Copy Constructor
   * @param gaussian_mixture
   */
  GaussianMixture(const GaussianMixture &gaussian_mixture) : weights_(gaussian_mixture.weights_), gaussians_(gaussian_mixture.gaussians_) {}

  /** Calculate the probability density function of x given the Gaussian mixture
   * @param x
   * @return double
   */
  double pdf(const Vec_n &x) const
  {
    double pdf = 0;
    for (int i = 0; i < num_components(); i++) {
      pdf += get_weight(i) * gaussian(i).pdf(x);
    }
    pdf /= sum_weights();
    return pdf;
  }

  /** Find the mean of the Gaussian mixture
   * @return Vector
   */
  Vec_n mean() const
  {
    Vec_n mean = Vec_n::Zero();
    for (size_t i = 0; i < gaussians_.size(); i++) {
      mean += get_weight(i) * gaussian(i).mean();
    }
    mean /= sum_weights();
    return mean;
  }

  /** Find the covariance of the Gaussian mixture
   * @return Matrix
   */
  Mat_nn cov() const
  {
    // Spread of innovations
    Mat_nn P_bar = Mat_nn::Zero();
    for (int i = 0; i < num_components(); i++) {
      P_bar += get_weight(i) * gaussian(i).mean() * gaussian(i).mean().transpose();
    }
    P_bar /= sum_weights();
    P_bar -= mean() * mean().transpose();

    // Spread of Gaussians
    Mat_nn P = Mat_nn::Zero();
    for (int i = 0; i < num_components(); i++) {
      P += get_weight(i) * gaussian(i).cov();
    }
    P /= sum_weights();
    return P + P_bar;
  }

  /** Reduce the Gaussian mixture to a single Gaussian
   * @return Gauss_n
   */
  Gauss_n reduce() const { return {mean(), cov()}; }

  /** Get the weights of the Gaussian mixture
   * @return Eigen::VectorXd
   */
  const Eigen::VectorXd &weights() const { return weights_; }

  /** Get the weight of the i'th Gaussian
   * @param i index
   * @return double
   */
  double get_weight(size_t i) const { return weights_(i); }

  /** Set the weight of the i'th Gaussian
   * @param i index
   * @param weight
   */
  void set_weight(size_t i, double weight) { weights_(i) = weight; }

  /** Get the sum of the weights
   * @return double
   */
  double sum_weights() const { return weights_.sum(); }

  /** Get the Gaussians of the Gaussian mixture
   * @return std::vector<Gauss_n>
   */
  const std::vector<Gauss_n> &gaussians() const { return gaussians_; }

  /** Get the i'th Gaussian
   * @param i index
   * @return Gauss_n
   */
  const Gauss_n &gaussian(size_t i) const { return gaussians_.at(i); }

  /** Get the number of Gaussians in the mixture
   * @return int
   */
  int num_components() const { return gaussians_.size(); }

  /** Sample from the Gaussian mixture
   * @param gen Random number generator
   * @return Vec_n
   */
  Vec_n sample(std::mt19937 &gen) const
  {
    std::discrete_distribution<int> dist(weights().begin(), weights().end());
    return gaussians_[dist(gen)].sample(gen);
  }

  /** Sample from the Gaussian mixture
   * @return Vec_n
   */
  Vec_n sample() const
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    return sample(gen);
  }

private:
  const Eigen::VectorXd weights_;
  const std::vector<Gauss_n> gaussians_;
};

template <int n_dims> using GaussMix = GaussianMixture<n_dims>;
using GaussMix2d                     = GaussMix<2>;
using GaussMix3d                     = GaussMix<3>;
using GaussMix4d                     = GaussMix<4>;

} // namespace prob
} // namespace vortex
