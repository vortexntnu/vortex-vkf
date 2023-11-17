#include <vortex_filtering/plotting/utils.hpp>

namespace vortex {
namespace plotting {

Ellipse gauss_to_ellipse(const vortex::prob::MultiVarGauss<2>& gauss)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(gauss.cov());
    Eigen::Vector2d eigenValues = eigenSolver.eigenvalues();
    Eigen::Vector2d eigenVectors = eigenSolver.eigenvectors();

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

}  // namespace plotting
}  // namespace vortex