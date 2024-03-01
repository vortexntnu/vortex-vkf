/**
 * @file ellipse.hpp
 * @author Eirik Kolås
 * @brief Ellipse class and ellipsoid class
 * @version 0.1
 * @date 2024-02-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once
#include <Eigen/Dense>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace utils {

/** Class for representing an ellipse.
 *
 */
class Ellipse {
public:
  /** Construct a new Ellipse object
   * @param center center of the ellipse
   * @param a half the length of the major axis (radius of the circumscribed circle)
   * @param b half the length of the minor axis (radius of the inscribed circle)
   * @param angle angle in radians
   */
  Ellipse(const Eigen::Vector2d &center, double a, double b, double angle);

  /** Construct a new Ellipse object from a Gaussian
   * @param gauss 2D Gaussian distribution
   * @param scale_factor scale factor for the ellipse
   */
  Ellipse(const vortex::prob::Gauss2d &gauss, double scale_factor = 1.0);

  /** Get the center of the ellipse
   * @return Eigen::Vector2d
   */
  Eigen::Vector2d center() const;

  /** Get x coordinate of the center
   * @return double
   */
  double x() const;

  /** Get y coordinate of the center
   * @return double
   */
  double y() const;

  /** Get the a parameter of the ellipse (half the length of the major axis)
   * @return double
   */
  double a() const;

  /** Get the b parameter of the ellipse (half the length of the minor axis)
   * @return double
   */
  double b() const;

  /** Get the major axis length of the ellipse
   * @return double
   */
  double major_axis() const;

  /** Get the minor axis length of the ellipse
   * @return double
   */
  double minor_axis() const;

  /** Get the axes lengths of the ellipse
   * @return Eigen::Vector2d
   */
  Eigen::Vector2d axes() const;

  /** Get the angle of the ellipse with respect to the x-axis
   * @return double
   */
  double angle_rad() const;

  /** Get the angle in degrees
   * @return double
   */
  double angle_deg() const;

private:
  Eigen::Vector2d center_;
  double a_;
  double b_;
  double angle_;
};

} // namespace utils
} // namespace vortex
