#include <gtest/gtest.h>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include "test_models.hpp"

namespace simple_sensor_model_test {

using SensorModel = vortex::models::IdentitySensorModel<2,1>;
using Measurement = typename SensorModel::Vec_z;
using Vec_x = typename SensorModel::Vec_x;
using Vec_z = typename SensorModel::Vec_z;
using Mat_xx = typename SensorModel::Mat_xx;
using Mat_zz = typename SensorModel::Mat_zz;
using Mat_zx = typename SensorModel::Mat_zx;
using Mat_xz = typename SensorModel::Mat_xz;
using Gauss_x = typename SensorModel::Gauss_x;
using Gauss_z = typename SensorModel::Gauss_z;


TEST(SensorModel, initSimpleModel)
{   
    SensorModel model(0.1);
    EXPECT_EQ(model.h(Vec_x::Zero()), Vec_z::Zero());

    Vec_x x{1,2};
    Vec_z z{1};
    EXPECT_EQ(model.h(x), z);
}

TEST(SensorModel, predictSimpleModel)
{
    SensorModel model(std::sqrt(0.1));
    Gauss_x x_est{Vec_x::Zero(), Mat_xx::Identity()};
    Gauss_z pred = model.pred_from_est(x_est);
    EXPECT_EQ(pred.mean(), Vec_z::Zero());
    EXPECT_TRUE(pred.cov().isApprox(Mat_zz::Identity()*1.1));

    pred = model.pred_from_state(x_est.mean());
    EXPECT_TRUE(pred.cov().isApprox(Mat_zz::Identity()*0.1));
}


} // simple_sensor_model_test
