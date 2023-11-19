#include <vortex_filtering/plotting/utils.hpp>

namespace vortex {
namespace plotting {

Ellipse gauss_to_ellipse(const vortex::prob::MultiVarGauss<2>& gauss)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(gauss.cov());
    Eigen::Vector2d eigenValues = eigenSolver.eigenvalues();
    Eigen::Matrix2d eigenVectors = eigenSolver.eigenvectors();

    double majorAxisLength = sqrt(eigenValues(1)); 
    double minorAxisLength = sqrt(eigenValues(0));
    double angle = atan2(eigenVectors(1, 1), eigenVectors(0, 1)) * 180.0 / M_PI; // Convert to degrees

    Ellipse ellipse;
    ellipse.x = gauss.mean()(0);
    ellipse.y = gauss.mean()(1);
    ellipse.a = majorAxisLength;
    ellipse.b = minorAxisLength;
    ellipse.angle = angle;
    return ellipse;
}

std::vector<double> create_nees_series(const std::vector<Eigen::VectorXd>& errors, 
                                       const std::vector<Eigen::MatrixXd>& covariances, 
                                       const std::vector<size_t>& indices) 
{
    std::vector<double> nees_series;

    for (size_t i = 0; i < errors.size(); ++i) {
        Eigen::VectorXd error = errors[i];
        Eigen::MatrixXd covariance = covariances[i];

        // Handle indices if provided
        if (!indices.empty()) {
            Eigen::VectorXd error_sub(indices.size());
            Eigen::MatrixXd covariance_sub(indices.size(), indices.size());

            for (size_t j = 0; j < indices.size(); ++j) {
                error_sub(j) = error(indices[j]);
                for (size_t k = 0; k < indices.size(); ++k) {
                    covariance_sub(j, k) = covariance(indices[j], indices[k]);
                }
            }

            error = error_sub;
            covariance = covariance_sub;
        }

        // NEES calculation
        double nees = error.transpose() * covariance.inverse() * error;
        nees_series.push_back(nees);
    }

    return nees_series;
}


std::vector<Eigen::VectorXd> create_error_series(const std::vector<Eigen::VectorXd>& x_true, const std::vector<vortex::prob::GaussXd>& x_est)
{
    std::vector<Eigen::VectorXd> error_series;
    for (size_t i = 0; i < x_true.size(); ++i) {
        error_series.push_back(x_true[i] - x_est[i].mean());
    }
    return error_series;
}

std::vector<double> extract_state_series(const std::vector<Eigen::VectorXd>& x_series, size_t index)
{
    std::vector<double> state_series;
    for (size_t i = 0; i < x_series.size(); ++i) {
        state_series.push_back(x_series[i](index));
    }
    return state_series;
}

std::vector<Eigen::VectorXd> extract_mean_series(const std::vector<vortex::prob::GaussXd>& x_series)
{
    std::vector<Eigen::VectorXd> mean_series;
    for (size_t i = 0; i < x_series.size(); ++i) {
        mean_series.push_back(x_series[i].mean());
    }
    return mean_series;
}

vortex::prob::GaussXd approximate_gaussian(const std::vector<Eigen::VectorXd>& samples)
{
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(samples[0].size());
    for (const auto& sample : samples) {
        mean += sample;
    }
    mean /= samples.size();

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(samples[0].size(), samples[0].size());
    for (const auto& sample : samples) {
        cov += (sample - mean) * (sample - mean).transpose();
    }
    cov /= samples.size();

    return {mean, cov};
}

}  // namespace plotting
}  // namespace vortex