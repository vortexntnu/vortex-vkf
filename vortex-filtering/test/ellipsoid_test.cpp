#include "gtest_assertions.hpp"

#include <gtest/gtest.h>
#include <numbers>
#include <vortex_filtering/utils/ellipse.hpp>
#include <vortex_filtering/utils/ellipsoid.hpp>

TEST(Ellipse, constructorFromParams)
{
  Eigen::Vector2d center(1.0, 2.0);
  double a     = 3.0;
  double b     = 2.0;
  double angle = std::numbers::pi / 2.0;
  Eigen::Matrix2d cov;
  vortex::utils::Ellipse ellipse(center, a, b, angle);

  EXPECT_EQ(ellipse.center(), center);
  EXPECT_EQ(ellipse.a(), a);
  EXPECT_EQ(ellipse.b(), b);
}

TEST(Ellipse, constructorFromGauss)
{
  Eigen::Vector2d center(1.0, 2.0);
  double a     = 3.0;
  double b     = 2.0;
  double angle = std::numbers::pi / 2.0;

  Eigen::Matrix2d eigenvectors{{std::cos(angle), -std::sin(angle)}, {std::sin(angle), std::cos(angle)}};
  Eigen::Vector2d eigenvalues = {a * a, b * b};

  Eigen::Matrix2d cov = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

  vortex::prob::Gauss2d gauss(center, cov);
  vortex::utils::Ellipse ellipse(gauss);

  EXPECT_EQ(ellipse.center(), center);
  EXPECT_EQ(ellipse.a(), a);
  EXPECT_EQ(ellipse.b(), b);
}

TEST(Ellipsoid, constructorFromGauss)
{
  Eigen::Vector2d center(1.0, 2.0);
  Eigen::Matrix2d cov{{1.0, 0.0}, {0.0, 9.0}};
  vortex::prob::Gauss2d gauss(center, cov);
  vortex::utils::Ellipsoid<2> ellipsoid(gauss);

  EXPECT_EQ(ellipsoid.center(), center);
  EXPECT_EQ(ellipsoid.semi_axis_lengths()(0), 3.0);
  EXPECT_EQ(ellipsoid.semi_axis_lengths()(1), 1.0);
  EXPECT_EQ(ellipsoid.volume(), std::numbers::pi * 3.0 * 1.0);
}

TEST(Ellipsoid, sameAsEllipse)
{
  Eigen::Vector2d center(1.0, 2.0);
  double a     = 3.0;
  double b     = 2.0;
  double angle = std::numbers::pi / 2.0;

  Eigen::Matrix2d eigenvectors{{std::cos(angle), -std::sin(angle)}, {std::sin(angle), std::cos(angle)}};
  Eigen::Vector2d eigenvalues = {a * a, b * b};

  Eigen::Matrix2d cov = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

  vortex::prob::Gauss2d gauss(center, cov);
  vortex::utils::Ellipsoid<2> ellipsoid(gauss);
  vortex::utils::Ellipse ellipse(gauss);

  EXPECT_EQ(ellipsoid.center(), ellipse.center());
  EXPECT_EQ(ellipsoid.axis_lengths(), ellipse.axes());
  EXPECT_EQ(ellipsoid.volume(), ellipse.area());
}