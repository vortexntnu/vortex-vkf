#include <gtest/gtest.h>
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/movement_models.hpp>
#include "test_models.hpp"

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

