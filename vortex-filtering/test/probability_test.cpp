#include <gtest/gtest.h>

#include <vortex_filtering/probability/binomial.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/probability/poisson.hpp>


TEST(multiVarGauss, initGaussian)
{
    vortex::prob::MultiVarGauss<2> gaussian(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());

    EXPECT_EQ(gaussian.mean(), (Eigen::Vector2d{0, 0}));
    EXPECT_EQ(gaussian.cov(), (Eigen::Matrix2d{{1, 0}, {0, 1}}));
}

TEST(multiVarGauss, pdf)
{
    vortex::prob::MultiVarGauss<2> gaussian(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());

    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 0}), 1/(2*M_PI), 1e-15);
    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 0}), 1/(2*std::sqrt(M_E)*M_PI), 1e-15);
    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 1}), 1/(2*std::sqrt(M_E)*M_PI), 1e-15);
    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 1}), 1/(2*M_E*M_PI), 1e-15);

    gaussian = vortex::prob::MultiVarGauss<2>(Eigen::Vector2d{0, 0}, Eigen::Matrix2d{{2, 1}, {1, 2}});

    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 0}), 1/(2*std::sqrt(3)*M_PI), 1e-15);
    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 0}), 1/(2*std::sqrt(3)*std::exp(1.0/3)*M_PI), 1e-15);
    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{0, 1}), 1/(2*std::sqrt(3)*std::exp(1.0/3)*M_PI), 1e-15);
    EXPECT_NEAR(gaussian.pdf(Eigen::Vector2d{1, 1}), 1/(2*std::sqrt(3)*std::exp(1.0/3)*M_PI), 1e-15);
}

TEST(multiVarGauss, invalidCovariance)
{
    EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()), std::invalid_argument);
    EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1, 0}, {0, 0}}), std::invalid_argument);
    EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{-1, 0}, {0, -1}}), std::invalid_argument);
}
