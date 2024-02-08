#include <vortex_filtering/plotting/utils.hpp>

namespace vortex {
namespace plotting {

Ellipse gauss_to_ellipse(const vortex::prob::Gauss2d &gauss)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(gauss.cov());
  Eigen::Vector2d eigenValues  = eigenSolver.eigenvalues();
  Eigen::Matrix2d eigenVectors = eigenSolver.eigenvectors();

  double majorAxisLength = sqrt(eigenValues(1));
  double minorAxisLength = sqrt(eigenValues(0));
  double angle           = atan2(eigenVectors(1, 1), eigenVectors(0, 1)) * 180.0 / M_PI; // Convert to degrees

  Ellipse ellipse;
  ellipse.x     = gauss.mean()(0);
  ellipse.y     = gauss.mean()(1);
  ellipse.a     = majorAxisLength;
  ellipse.b     = minorAxisLength;
  ellipse.angle = angle;
  return ellipse;
}

} // namespace plotting
} // namespace vortex