#pragma once
#include <cmath>

namespace vortex::prob {

class ChiSquared {
public:
  ChiSquared(int n) : n_(n) {}

  /** Calculate the probability density of x given the Chi-Squared distribution
   * @param x
   * @return double
   */
  double pdf(double x) const { return std::pow(x, n_ / 2 - 1) * std::exp(-x / 2) / (std::pow(2, n_ / 2) * std::tgamma(n_ / 2)); }

  /** Calculate the mean of the Chi-Squared distribution
   * @return double mean
   */
  double mean() const { return n_; }

  /** Calculate the variance of the Chi-Squared distribution
   * @return double variance
   */
  double cov() const { return 2 * n_; }

  /** Parameter n of the Chi-Squared distribution
   * @return int n
   */
  int n() const { return n_; }

private:
  int n_;
};

} // namespace vortex::prob
