/**
 * @file poisson.hpp
 * @author Eirik Kolås
 * @brief A class for representing a Poisson distribution. Used for modeling clutter
 * @version 0.1
 * @date 2023-10-26
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include <cmath>

namespace vortex::prob {

class Poisson {
public:
  Poisson(double lambda) : lambda_(lambda) {}

  /** Calculate the probability of x given the Poisson distribution
   * @param x
   * @return double
   */
  double pr(int x) const { return std::pow(lambda_, x) * std::exp(-lambda_) / factorial(x); }

  /** Calculate the mean of the Poisson distribution
   * @return double mean
   */
  double mean() const { return lambda_; }

  /** Calculate the variance of the Poisson distribution
   * @return double variance
   */
  double cov() const { return lambda_; }

  /** Parameter lambda of the Poisson distribution
   * @return double lambda
   */
  double lambda() const { return lambda_; }

private:
  const double lambda_;

  /** Calculate the factorial of x
   * @param x
   * @return double factorial
   */
  static constexpr double factorial(int x)
  {
    return (x == 1 || x == 0) ? 1 : factorial(x - 1) * x;
  }
};

} // namespace vortex::prob
