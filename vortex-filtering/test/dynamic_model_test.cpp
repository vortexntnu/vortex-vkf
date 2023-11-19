#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <gnuplot-iostream.h>

#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/plotting/utils.hpp>

#include "test_models.hpp"
#include "gtest_assertions.hpp"

namespace simple_dynamic_model_test {

using Vec_x  = typename SimpleDynamicModel::Vec_x;
using Mat_xx = typename SimpleDynamicModel::Mat_xx;

TEST(DynamicModel, initSimpleModel)
{   
    SimpleDynamicModel model;
}

TEST(DynamicModel, iterateSimpleModel)
{
    SimpleDynamicModel model;
    double dt = 1.0;
    Vec_x x = Vec_x::Zero();

    for (size_t i = 0; i < 10; i++)
    {
        EXPECT_EQ(model.f_d(x, dt), std::exp(-dt) * x);
        x = model.f_d(x, dt);
    }

}

TEST(DynamicModel, sampleSimpleModel)
{
    // Test that output is Gaussian
    SimpleDynamicModel model;
    double dt = 1.0;
    Vec_x x = Vec_x::Ones();

    vortex::prob::Gauss2d true_gauss = model.pred_from_state(x, dt);

    std::random_device rd;                            
    std::mt19937 gen(rd());   

    std::vector<Eigen::VectorXd> samples;
    for (size_t i = 0; i < 10000; i++)
    {
        samples.push_back(model.sample_f_d(x, dt, gen));
    }

    vortex::prob::Gauss2d approx_gauss = vortex::plotting::approximate_gaussian(samples);

    EXPECT_TRUE(isApproxEqual(approx_gauss.mean(), true_gauss.mean(), 0.1));
    EXPECT_TRUE(isApproxEqual(true_gauss.cov(), true_gauss.cov(), 0.1));

    // Plot

    Gnuplot gp;
    gp << "set xrange [-10:10]\nset yrange [-10:10]\n";
    gp << "set style circle radius 0.05\n";
    gp << "plot '-' with circles title 'Samples' fs transparent solid 0.05 noborder\n";
    gp.send1d(samples);

    vortex::plotting::Ellipse cov_ellipse = vortex::plotting::gauss_to_ellipse(true_gauss);
    gp << "set object 1 ellipse center " << cov_ellipse.x << "," << cov_ellipse.y 
       << " size " << 3*cov_ellipse.a << "," << 3*cov_ellipse.b 
       << " angle " << cov_ellipse.angle << "fs empty border lc rgb 'cyan'\n";
    gp << "replot\n";

}



} // namespace simple_model_test

namespace cv_model_test {

using Vec_x  = typename vortex::models::CVModel::Vec_x;
using Mat_xx = typename vortex::models::CVModel::Mat_xx;

TEST(DynamicModel, initCVModel)
{   
    vortex::models::CVModel model(1.0);
}

TEST(DynamicModel, iterateCVModel)
{
    vortex::models::CVModel model(1.0);
    double dt = 1.0;
    Vec_x x;
    x << 0, 0, 1, 1;

    for (size_t i = 0; i < 10; i++)
    {
        Vec_x x_true;
        x_true << x(0) + dt, x(1) + dt, 1, 1;
        EXPECT_EQ(model.f_d(x, dt), x_true);
        x = model.f_d(x, dt);
    }

}



} // namespace cv_model_test

