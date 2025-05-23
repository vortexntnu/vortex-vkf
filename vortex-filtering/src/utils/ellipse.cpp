#include <numbers>
#include <vortex_filtering/utils/ellipse.hpp>

using std::numbers::pi;

namespace vortex {
namespace utils {

Ellipse::Ellipse(const Eigen::Vector2d& center,
                 double a,
                 double b,
                 double angle)
    : center_(center), a_(a), b_(b), angle_(angle) {}

Ellipse::Ellipse(const vortex::prob::Gauss2d& gauss, double scale_factor) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(gauss.cov());
    Eigen::Vector2d eigenValues = eigenSolver.eigenvalues();
    Eigen::Matrix2d eigenVectors = eigenSolver.eigenvectors();

    // Sort eigenvalues in descending order
    if (eigenValues(0) > eigenValues(1)) {
        std::swap(eigenValues(0), eigenValues(1));
        eigenVectors.col(0).swap(eigenVectors.col(1));
    }

    a_ = sqrt(scale_factor * eigenValues(1));
    b_ = sqrt(scale_factor * eigenValues(0));
    angle_ = atan2(eigenVectors(1, 1), eigenVectors(0, 1));
    center_ = gauss.mean();
}

Eigen::Vector2d Ellipse::center() const {
    return center_;
}
double Ellipse::x() const {
    return center_(0);
}
double Ellipse::y() const {
    return center_(1);
}
double Ellipse::a() const {
    return a_;
}
double Ellipse::b() const {
    return b_;
}
double Ellipse::major_axis() const {
    return 2 * a_;
}
double Ellipse::minor_axis() const {
    return 2 * b_;
}
Eigen::Vector2d Ellipse::axes() const {
    return Eigen::Vector2d(2 * a_, 2 * b_);
}
double Ellipse::angle_rad() const {
    return angle_;
}
double Ellipse::angle_deg() const {
    return angle_ * 180 / std::numbers::pi;
}
double Ellipse::area() const {
    return std::numbers::pi * a() * b();
}

}  // namespace utils
}  // namespace vortex
