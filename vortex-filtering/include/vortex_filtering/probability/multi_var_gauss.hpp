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
	using Vector = Eigen::Vector<double, n_dims>;
	using Matrix = Eigen::Matrix<double, n_dims, n_dims>;

	/** Construct a Gaussian with a given mean and covariance matrix
	 * @param mean
	 * @param cov Symmetric positive definite covariance matrix
	 */
	MultiVarGauss(const Vector &mean, const Matrix &cov)
	    : N_DIMS(mean.size()), mean_(mean), cov_(cov), cov_inv_(cov_.llt().solve(Matrix::Identity(size(), size())))
	{
		// Check that the covariance matrix is positive definite and symmetric
		if (!cov_.isApprox(cov_.transpose(), 1e-6)) {
			throw std::invalid_argument("Covariance matrix is not symmetric");
		}
		if (cov_.llt().info() != Eigen::Success) {
			throw std::invalid_argument("Covariance matrix is not positive definite");
		}
	}

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

		Vector mean = other.mean();
		Matrix cov  = other.cov();

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
	double pdf(const Vector &x) const
	{
		const Vector diff     = x - mean();
		const double exponent = -0.5 * diff.transpose() * cov_inv() * diff;
		return std::exp(exponent) / std::sqrt(std::pow(2 * M_PI, size()) * cov().determinant());
	}

	/** Calculate the log likelihood of x given the Gaussian.
	 * Assumes that the covariance matrix is positive definite and symmetric
	 * @param x
	 * @return double
	 */
	double logpdf(const Vector &x) const
	{
		const Vector diff     = x - mean();
		const double exponent = -0.5 * diff.transpose() * cov_inv() * diff;
		return exponent - 0.5 * std::log(std::pow(2 * M_PI, size()) * cov().determinant());
	}

	Vector mean() const { return mean_; }
	Matrix cov() const { return cov_; }
	Matrix cov_inv() const { return cov_inv_; }

	/** Calculate the Mahalanobis distance of x given the Gaussian
	 * @param x
	 * @return double
	 */
	double mahalanobis_distance(const Vector &x) const
	{
		const auto diff = x - mean();
		return std::sqrt(diff.transpose() * cov_inv() * diff);
	}

	/** Sample from the Gaussian
	 * @param gen Random number generator
	 * @return Vector
	 */
	Vector sample(std::mt19937 &gen) const
	{
		std::normal_distribution<> d{0, 1};
		Vector sample(size());
		for (int i = 0; i < size(); ++i) {
			sample(i) = d(gen);
		}
		return mean() + cov().llt().matrixL() * sample;
	}

	/** Sample from the Gaussian
	 * @return Vector
	 */
	Vector sample() const
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		return sample(gen);
	}

	/** size of the Gaussian
	 * @return int
	 */
	int size() const { return N_DIMS; }

private:
	size_t N_DIMS;
	Vector mean_;
	Matrix cov_;
	Matrix cov_inv_;
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
