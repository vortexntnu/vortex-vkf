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
#include <vector>
#include <numeric> // std::accumulate
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace prob {

/**
 * A class for representing a multivariate Gaussian mixture distribution
 * @tparam N_DIM_x dimentions of the Gaussian
 */
template <int N_DIM_x> class GaussianMixture {
public:
	using Vec = Eigen::Vector<double, N_DIM_x>;
	using Mat = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;

	/** Construct a new Gaussian Mixture object
	 * @param weights Weights of the Gaussians
	 * @param gaussians Gaussians
	 * @note The weights are automatically normalized, so they do not need to sum to 1.
	 */
	GaussianMixture(std::vector<double> weights, std::vector<MultiVarGauss<N_DIM_x>> gaussians) : weights_(std::move(weights)), gaussians_(std::move(gaussians)) {}

	/** Calculate the probability density function of x given the Gaussian mixture
	 * @param x
	 * @return double
	 */
	double pdf(const Vec &x) const
	{
		double pdf = 0;
		for (int i = 0; i < gaussians_.size(); i++) {
			pdf += weights_[i] * gaussians_[i].pdf(x);
		}
		pdf /= sum_weights();
		return pdf;
	}

	/** Find the mean of the Gaussian mixture
	 * @return Vector
	 */
	Vec mean() const
	{
		Vec mean = Vec::Zero();
		for (int i = 0; i < gaussians_.size(); i++) {
			mean += weights_[i] * gaussians_[i].mean();
		}
		mean /= sum_weights();
		return mean;
	}

	/** Find the covariance of the Gaussian mixture
	 * @return Matrix
	 */
	Mat cov() const
	{
		// Spread of innovations
		Mat P_bar = Mat::Zero();
		for (int i = 0; i < gaussians_.size(); i++) {
			P_bar += weights_[i] * gaussians_[i].mean() * gaussians_[i].mean().transpose();
		}
		P_bar /= sum_weights();
		P_bar -= mean() * mean().transpose();

		// Spread of Gaussians
		Mat P = Mat::Zero();
		for (int i = 0; i < gaussians_.size(); i++) {
			P += weights_[i] * gaussians_[i].cov();
		}
		P /= sum_weights();
		return P + P_bar;
	}

	/** Reduce the Gaussian mixture to a single Gaussian
	 * @return MultiVarGauss
	 */
	MultiVarGauss<N_DIM_x> reduce() const { return MultivarGauss(mean(), cov()); }

	/** dimentions of the Gaussian mixture
	 * @return int
	 */
	int n_dims() const { return (N_DIM_x); }

	/** Get the weights of the Gaussian mixture
	 * @return std::vector<int>
	 */
	std::vector<double> weights() const { return weights_; }

	/** Get the Gaussians of the Gaussian mixture
	 * @return std::vector<MultiVarGauss<N_DIM_x>>
	 */
	std::vector<MultiVarGauss<N_DIM_x>> gaussians() const { return gaussians_; }


	/** Sample from the Gaussian mixture
	 * @param gen Random number generator
	 * @return Vec
	 */
	Vec sample(std::mt19937 &gen) const
	{
		std::discrete_distribution<int> dist(weights_.begin(), weights_.end());
		return gaussians_[dist(gen)].sample(gen);
	}

	/** Sample from the Gaussian mixture
	 * @return Vec
	 */
	Vec sample() const
	{
		std::random_device rd;
		std::mt19937       gen(rd());
		return sample(gen);
	}

private:
	const std::vector<double> weights_;
	const std::vector<MultiVarGauss<N_DIM_x>> gaussians_;

	double sum_weights() const { return std::accumulate(weights_.begin(), weights_.end(), 0.0); }
};

template <int N_DIM_x>
GaussianMixture<N_DIM_x> operator+(const GaussianMixture<N_DIM_x> &lhs, const GaussianMixture<N_DIM_x> &rhs)
{
	std::vector<double> weights = lhs.weights();
	weights.insert(weights.end(), rhs.weights().begin(), rhs.weights().end());
	std::vector<MultiVarGauss<N_DIM_x>> gaussians = lhs.gaussians();
	gaussians.insert(gaussians.end(), rhs.gaussians().begin(), rhs.gaussians().end());
	return GaussianMixture<N_DIM_x>(weights, gaussians);
}

template <int N_DIM_x>
GaussianMixture<N_DIM_x> operator*(double weight, const GaussianMixture<N_DIM_x> &rhs)
{
	std::vector<double> weights = rhs.weights();
	for (int i = 0; i < weights.size(); i++) {
		weights[i] *= weight;
	}
	return GaussianMixture<N_DIM_x>(weights, rhs.gaussians());
}

template <int N_DIM_x>
GaussianMixture<N_DIM_x> operator*(const GaussianMixture<N_DIM_x> &lhs, double weight)
{
	return weight * lhs;
}

template <int N_DIM_x>
GaussianMixture<N_DIM_x> operator*(double weight, const MultiVarGauss<N_DIM_x> &rhs)
{
	return weight * GaussianMixture<N_DIM_x>({1}, {rhs});
}

template <int N_DIM_x>
GaussianMixture<N_DIM_x> operator*(const MultiVarGauss<N_DIM_x> &lhs, double weight)
{
	return weight * lhs;
}

} // namespace prob
} // namespace vortex
