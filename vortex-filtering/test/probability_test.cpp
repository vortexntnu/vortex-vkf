#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <random>

#include <vortex_filtering/probability/binomial.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/probability/poisson.hpp>
#include <vortex_filtering/utils/ellipse.hpp>

#include "gtest_assertions.hpp"

TEST(MultiVarGauss, initGaussian)
{
  vortex::prob::MultiVarGauss<2> gaussian(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());

  EXPECT_EQ(gaussian.mean(), (Eigen::Vector2d{0, 0}));
  EXPECT_EQ(gaussian.cov(), (Eigen::Matrix2d{{1, 0}, {0, 1}}));
}

TEST(MultiVarGauss, assignmentOperator)
{
  vortex::prob::MultiVarGauss<2> gaussian1(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
  vortex::prob::MultiVarGauss<2> gaussian2(Eigen::Vector2d::Ones(), Eigen::Matrix2d::Identity());

  gaussian2 = gaussian1;

  EXPECT_EQ(gaussian2.mean(), gaussian1.mean());
  EXPECT_EQ(gaussian2.cov(), gaussian1.cov());
}

TEST(MultiVarGauss, copyConstructor)
{
  vortex::prob::MultiVarGauss<2> gaussian1(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());
  vortex::prob::MultiVarGauss<2> gaussian2(gaussian1);

  EXPECT_EQ(gaussian2.mean(), gaussian1.mean());
  EXPECT_EQ(gaussian2.cov(), gaussian1.cov());
}

TEST(MultiVarGauss, pdf)
{
  vortex::prob::MultiVarGauss<2> gaussian(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());

  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 0}), 1 / (2 * M_PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 0}), 1 / (2 * std::sqrt(M_E) * M_PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 1}), 1 / (2 * std::sqrt(M_E) * M_PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 1}), 1 / (2 * M_E * M_PI), 1e-15);

  gaussian = vortex::prob::MultiVarGauss<2>(Eigen::Vector2d{0, 0}, Eigen::Matrix2d{{2, 1}, {1, 2}});

  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 0}), 1 / (2 * std::sqrt(3) * M_PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 0}), 1 / (2 * std::sqrt(3) * std::exp(1.0 / 3) * M_PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 1}), 1 / (2 * std::sqrt(3) * std::exp(1.0 / 3) * M_PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 1}), 1 / (2 * std::sqrt(3) * std::exp(1.0 / 3) * M_PI), 1e-15);
}

TEST(MultiVarGauss, invalidCovariance)
{
  EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()), std::invalid_argument);
  EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1, 0}, {0, 0}}), std::invalid_argument);
  EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{-1, 0}, {0, -1}}), std::invalid_argument);
}

TEST(MultiVarGauss, sample)
{
  Eigen::Vector2d true_mean{4, -2};
  Eigen::Matrix2d true_cov = Eigen::Matrix2d{{4, 1}, {1, 2}};

  vortex::prob::MultiVarGauss<2> gaussian(true_mean, true_cov);
  std::random_device rd;
  std::mt19937 gen(rd());

  std::vector<Eigen::Vector2d> samples;
  for (size_t i = 0; i < 10000; ++i) {
    samples.push_back(gaussian.sample(gen));
  }

  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for (const auto &sample : samples) {
    mean += sample;
  }
  mean /= samples.size();

  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto &sample : samples) {
    cov += (sample - mean) * (sample - mean).transpose();
  }
  cov /= samples.size();

  vortex::utils::Ellipse cov_ellipse({true_mean, true_cov});

  double majorAxisLength = cov_ellipse.major_axis();
  double minorAxisLength = cov_ellipse.minor_axis();
  double angle           = cov_ellipse.angle_deg();

  #ifdef GNUPLOT_ENABLE
  Gnuplot gp;
  gp << "set xrange [-10:10]\nset yrange [-10:10]\n";
  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' fs transparent solid 0.05 noborder\n";
  gp.send1d(samples);

  gp << "set object 1 ellipse center " << true_mean(0) << "," << true_mean(1) << " size " << majorAxisLength << "," << minorAxisLength << " angle " << angle
     << "fs empty border lc rgb 'cyan'\n";
  gp << "replot\n";
  #endif

  EXPECT_TRUE(isApproxEqual(mean, true_mean, 0.5));
  EXPECT_TRUE(isApproxEqual(cov, true_cov, 0.5));

  (void)majorAxisLength;
  (void)minorAxisLength;
  (void)angle;
}

TEST(MultiVarGauss, mahalanobisDistanceIdentityCovariance)
{
  using vortex::prob::Gauss2d;
  auto gaussian = Gauss2d::Standard();

  EXPECT_DOUBLE_EQ(gaussian.mahalanobis_distance({0, 0}), 0);
  EXPECT_DOUBLE_EQ(gaussian.mahalanobis_distance({1, 0}), 1);
  EXPECT_DOUBLE_EQ(gaussian.mahalanobis_distance({0, 1}), 1);
  EXPECT_DOUBLE_EQ(gaussian.mahalanobis_distance({1, 1}), std::sqrt(2));
}