/**
 * @file binomial.hpp
 * @author Eirik Kolås
 * @brief
 * @version 0.1
 * @date 2023-10-26
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include <cmath>

namespace vortex {
namespace prob {

class Binomial {
public:
	Binomial(int n, double p) : n_(n), p_(p) {}

	/** Calculate the probability of x given the Binomial distribution
	 * @param x
	 * @return double
	 */
	double pr(int x) const { return std::pow(p_, x) * std::pow(1 - p_, n_ - x) * factorial(n_) / (factorial(x) * factorial(n_ - x)); }

	/** Calculate the mean of the Binomial distribution
	 * @return double mean
	 */
	double mean() const { return n_ * p_; }

	/** Calculate the variance of the Binomial distribution
	 * @return double variance
	 */
	double cov() const { return n_ * p_ * (1 - p_); }

	/** Parameter n of the Binomial distribution
	 * @return int n
	 */
	int n() const { return n_; }

	/** Parameter p of the Binomial distribution
	 * @return double p
	 */
	double p() const { return p_; }

private:
	int n_;
	double p_;

	/** Calculate the factorial of x
	 * @param x
	 * @return double factorial
	 */
	double factorial(int x) const
	{
		double factorial = 1;
		for (int i = 1; i <= x; i++) {
			factorial *= i;
		}
		return factorial;
	}
};

} // namespace prob
} // namespace vortex
