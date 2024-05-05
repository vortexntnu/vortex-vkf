#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <random>
#include <numbers>

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
  constexpr double PI = std::numbers::pi;
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 0}), 1 / (2 * PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 0}), 1 / (2 * std::sqrt(M_E) * PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 1}), 1 / (2 * std::sqrt(M_E) * PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 1}), 1 / (2 * M_E * PI), 1e-15);

  gaussian = vortex::prob::MultiVarGauss<2>(Eigen::Vector2d{0, 0}, Eigen::Matrix2d{{2, 1}, {1, 2}});

  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 0}), 1 / (2 * std::sqrt(3) * PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 0}), 1 / (2 * std::sqrt(3) * std::exp(1.0 / 3) * PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 1}), 1 / (2 * std::sqrt(3) * std::exp(1.0 / 3) * PI), 1e-15);
  EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 1}), 1 / (2 * std::sqrt(3) * std::exp(1.0 / 3) * PI), 1e-15);
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

  #if (GNUPLOT_ENABLE)
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

TEST(isContainerConcept, compileTimeChecks)
{
  static_assert(vortex::prob::concepts::is_container<std::vector<double>, double>);
  static_assert(vortex::prob::concepts::is_container<std::array<double, 4>, double>);
  static_assert(vortex::prob::concepts::is_container<std::vector<vortex::prob::Gauss2d>, vortex::prob::Gauss2d>);
  static_assert(vortex::prob::concepts::is_container<Eigen::Vector2d, double>);
  static_assert(vortex::prob::concepts::is_container<Eigen::VectorXd, double>);
  static_assert(vortex::prob::concepts::is_container<Eigen::RowVectorXd, double>);

  static_assert(!vortex::prob::concepts::is_container<double, double>);

  EXPECT_TRUE(true);
}

TEST(GaussianMixture, defaultConstructor)
{
  vortex::prob::GaussianMixture<2> mixture;

  EXPECT_EQ(mixture.size(), 0u);
}

TEST(GaussianMixture, stdVectorConstructor)
{
  using vortex::prob::Gauss2d;
  std::vector<double> weights{1, 2};
  std::vector<Gauss2d> gaussians{Gauss2d::Standard(), Gauss2d::Standard()};

  vortex::prob::GaussianMixture<2> mixture{weights, gaussians};

  EXPECT_EQ(mixture.size(), 2u);

  Eigen::VectorXd weights_eigen(2);
  weights_eigen << 1, 2;
  EXPECT_EQ(mixture.weights(), weights_eigen);
  EXPECT_EQ(mixture.gaussians(), gaussians);
}

TEST(GaussianMixture, stdArrayConstructor)
{
  using vortex::prob::Gauss2d;
  std::array<double, 2> weights{1, 2};
  std::array<Gauss2d, 2> gaussians{Gauss2d::Standard(), Gauss2d::Standard()};

  vortex::prob::GaussianMixture<2> mixture{weights, gaussians};

  EXPECT_EQ(mixture.size(), 2u);

  Eigen::VectorXd weights_eigen(2);
  weights_eigen << 1, 2;
  EXPECT_EQ(mixture.weights(), weights_eigen);
  EXPECT_EQ(mixture.gaussians().at(0), gaussians.at(0));
  EXPECT_EQ(mixture.gaussians().at(1), gaussians.at(1));
}

TEST(GaussianMixture, eigenVectorConstructor)
{
  using vortex::prob::Gauss2d;
  Eigen::VectorXd weights(2);
  weights << 1, 2;
  std::vector<Gauss2d> gaussians{Gauss2d::Standard(), Gauss2d::Standard()};

  vortex::prob::GaussianMixture<2> mixture{weights, gaussians};

  EXPECT_EQ(mixture.size(), 2u);
  EXPECT_EQ(mixture.weights(), weights);
  EXPECT_EQ(mixture.gaussians().at(0), gaussians.at(0));
  EXPECT_EQ(mixture.gaussians().at(1), gaussians.at(1));
}

TEST(GaussianMixture, mixTwoEqualWeightEqualCovarianceComponents)
{
  using vortex::prob::Gauss2d;
  std::vector<double> weights{0.5, 0.5};

  Gauss2d gaussian1{{0, 0}, Eigen::Matrix2d::Identity()};
  Gauss2d gaussian2{{10, 0}, Eigen::Matrix2d::Identity()};
  std::vector<Gauss2d> gaussians{gaussian1, gaussian2};

  Eigen::Vector2d center{5, 0};

  vortex::prob::GaussianMixture<2> mixture{weights, gaussians};

  EXPECT_TRUE(isApproxEqual(mixture.reduce().mean(), center, 1e-15));
}

TEST(GaussianMixture, mixTwoEqualWeightDifferentCovarianceComponents)
{
  using vortex::prob::Gauss2d;
  std::vector<double> weights{0.5, 0.5};

  Gauss2d gaussian1{{0, 0}, Eigen::Matrix2d::Identity()};
  Gauss2d gaussian2{{10, 0}, Eigen::Matrix2d::Identity() * 2};
  std::vector<Gauss2d> gaussians{gaussian1, gaussian2};

  vortex::prob::GaussianMixture<2> mixture{weights, gaussians};

  Eigen::Vector2d center{5, 0};

  EXPECT_TRUE(isApproxEqual(mixture.reduce().mean(), center, 1e-15));
}

TEST(GaussianMixture, mixTwoDifferentWeightEqualCovarianceComponents)
{
  using vortex::prob::Gauss2d;
  std::vector<double> weights{0.25, 0.75};

  Gauss2d gaussian1{{0, 0}, Eigen::Matrix2d::Identity()};
  Gauss2d gaussian2{{10, 0}, Eigen::Matrix2d::Identity()};
  std::vector<Gauss2d> gaussians{gaussian1, gaussian2};

  vortex::prob::GaussianMixture<2> mixture{weights, gaussians};

  Eigen::Vector2d center{7.5, 0};

  EXPECT_TRUE(isApproxEqual(mixture.reduce().mean(), center, 1e-15));
}
