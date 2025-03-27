#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <numbers>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::utils {

template <size_t n_dims>
class Ellipsoid {
   public:
    using T = Types_n<n_dims>;

    /** Construct a new Ellipsoid object
     * @param gauss n-dimensional Gaussian distribution
     * @param scale_factor scale factor for the number of standard deviations
     * (NB! This is slightly different from the ellipse scale factor)
     */
    Ellipsoid(T::Gauss_n gauss, double scale_factor = 1.0)
        : gauss_(gauss), scale_factor_(scale_factor) {}

    /** Get the center of the ellipsoid
     * @return T::Vec_n
     */
    T::Vec_n center() const { return gauss_.mean(); }

    /** Get the semi-major axis lengths of the ellipsoid sorted in descending
     * order
     * @return T::Vec_n
     */
    T::Vec_n semi_axis_lengths() const {
        Eigen::SelfAdjointEigenSolver<typename T::Mat_nn> eigen_solver(
            gauss_.cov());
        typename T::Vec_n eigenvalues = eigen_solver.eigenvalues();
        typename T::Vec_n lengths = eigenvalues.array().sqrt().reverse();
        return lengths * scale_factor_;
    }

    /** Get the axis lengths of the ellipsoid sorted in descending order
     * @return T::Vec_n
     */
    T::Vec_n axis_lengths() const { return 2.0 * semi_axis_lengths(); }

    /** Get the orthonormal axes of the ellipsoid corresponding to the eigen
     * values sorted in descending order
     * @return T::Mat_nn
     */
    T::Mat_nn orthonormal_axes() const {
        Eigen::SelfAdjointEigenSolver<typename T::Mat_nn> eigen_solver(
            gauss_.cov());
        typename T::Mat_nn eigen_vectors =
            eigen_solver.eigenvectors().colwise().reverse();
        return eigen_vectors;
    }

    /** Get the volume of the ellipsoid
     * @return double
     */
    double volume() const {
        constexpr double scaling = std::pow(std::numbers::pi, n_dims / 2.0) /
                                   std::tgamma(n_dims / 2.0 + 1);
        typename T::Vec_n lengths = semi_axis_lengths();
        double volume = scaling * lengths.prod();
        return volume;
    }

   private:
    typename T::Gauss_n gauss_;
    double scale_factor_;
};

}  // namespace vortex::utils
