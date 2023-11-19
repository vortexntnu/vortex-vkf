#include <gtest/gtest.h>
#include <random>
#include <gnuplot-iostream.h>

#include <vortex_filtering/probability/binomial.hpp>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/probability/poisson.hpp>

#include "gtest_assertions.hpp"

TEST(MultiVarGauss, initGaussian)
{
    vortex::prob::MultiVarGauss<2> gaussian(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity());

    EXPECT_EQ(gaussian.mean(), (Eigen::Vector2d{0, 0}));
    EXPECT_EQ(gaussian.cov(), (Eigen::Matrix2d{{1, 0}, {0, 1}}));
}

TEST(MultiVarGauss, pdf)
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

TEST(MultiVarGauss, invalidCovariance)
{
    EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Zero()), std::invalid_argument);
    EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{1, 0}, {0, 0}}), std::invalid_argument);
    EXPECT_THROW(vortex::prob::MultiVarGauss<2>(Eigen::Vector2d::Zero(), Eigen::Matrix2d{{-1, 0}, {0, -1}}), std::invalid_argument);
}

TEST(MultiVarGauss, initDynamicSize)
{
    vortex::prob::MultiVarGauss<-1> gaussian(Eigen::Vector2d{0, 3}, Eigen::Matrix2d{{4, 0}, {0, 1}});

    EXPECT_EQ(gaussian.mean(), (Eigen::Vector2d{0, 3}));
    EXPECT_EQ(gaussian.cov(), (Eigen::Matrix2d{{4, 0}, {0, 1}}));
}

TEST(MultiVarGauss, castToDynamicSize)
{
    vortex::prob::MultiVarGauss<2> gaussian(Eigen::Vector2d{0, 1}, Eigen::Matrix2d::Identity());
    vortex::prob::MultiVarGauss<-1> dynamic_gaussian = gaussian;

    EXPECT_EQ(dynamic_gaussian.mean(), (Eigen::Vector2d{0, 1}));
    EXPECT_EQ(dynamic_gaussian.cov(), (Eigen::Matrix2d{{1, 0}, {0, 1}}));
}

TEST(MultiVarGauss, castToStaticSize)
{
    vortex::prob::MultiVarGauss<-1> dynamic_gaussian(Eigen::Vector2d{2, 3}, Eigen::Matrix2d::Identity());
    vortex::prob::MultiVarGauss<2> gaussian = dynamic_gaussian;

    EXPECT_EQ(gaussian.mean(), (Eigen::Vector2d{2, 3}));
    EXPECT_EQ(gaussian.cov(), (Eigen::Matrix2d{{1, 0}, {0, 1}}));

    // Expect fail when trying to cast to wrong size
    vortex::prob::MultiVarGauss3d gaussian3 = {Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()};
    EXPECT_THROW(gaussian3 = dynamic_gaussian, std::invalid_argument);
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
    for (const auto& sample : samples) {
        mean += sample;
    }
    mean /= samples.size();

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (const auto& sample : samples) {
        cov += (sample - mean) * (sample - mean).transpose();
    }
    cov /= samples.size();

    // Plot points and an ellipse for the true mean and covariance. use the gp ellipse function
    double num_std_dev = 3.0; // Number of standard deviations to plot the ellipse at
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(true_cov);
    auto eigenValues = eigenSolver.eigenvalues();
    auto eigenVectors = eigenSolver.eigenvectors();

    double majorAxisLength = sqrt(eigenValues(1)) * num_std_dev; 
    double minorAxisLength = sqrt(eigenValues(0)) * num_std_dev;
    double angle = atan2(eigenVectors(1, 1), eigenVectors(0, 1)) * 180.0 / M_PI; // Convert to degrees


    Gnuplot gp;
    gp << "set xrange [-10:10]\nset yrange [-10:10]\n";
    gp << "set style circle radius 0.05\n";
    gp << "plot '-' with circles title 'Samples' fs transparent solid 0.05 noborder\n";
    gp.send1d(samples);

    gp << "set object 1 ellipse center " << true_mean(0) << "," << true_mean(1) 
       << " size " << majorAxisLength << "," << minorAxisLength 
       << " angle " << angle << "fs empty border lc rgb 'cyan'\n";
    gp << "replot\n";




    EXPECT_TRUE(isApproxEqual(mean, true_mean, 0.5));
    EXPECT_TRUE(isApproxEqual(cov, true_cov, 0.5));

}

