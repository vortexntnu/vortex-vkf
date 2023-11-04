#include <gtest/gtest.h>
#include <models/sensor_model.hpp>
#include <probability/multi_var_gauss.hpp>
#include "test_models.hpp"

namespace simple_sensor_model_test {

using Measurement = typename SimpleSensorModel::Vec_z;
using Vec_x = typename SimpleSensorModel::Vec_x;
using Mat_xx = typename SimpleSensorModel::Mat_xx;


TEST(SensorModel, initSimpleModel)
{   
    SimpleSensorModel model;
    EXPECT_EQ(model.h(Vec_x::Zero()), Vec_x::Zero());

    Vec_x x{1,2};
    EXPECT_EQ(model.h(x), x);
}

TEST(SensorModel, predictSimpleModel)
{
    SimpleSensorModel model;
    vortex::prob::MultiVarGauss<2> x_est{Vec_x::Zero(), Mat_xx::Identity()};
    vortex::prob::MultiVarGauss<2> pred = model.pred_from_est(x_est);
    EXPECT_EQ(pred.mean(), Vec_x::Zero());
    EXPECT_TRUE(pred.cov().isApprox(Mat_xx::Identity()*1.1));

    pred = model.pred_from_state(x_est.mean());
    EXPECT_TRUE(pred.cov().isApprox(Mat_xx::Identity()*0.1));
}


} // simple_sensor_model_test

namespace variable_length_sensor_model_test {

using Measurement = typename VariableLengthSensorModel::Vec_z;
using Vec_x = typename VariableLengthSensorModel::Vec_x;
using Mat_xx = typename VariableLengthSensorModel::Mat_xx;

TEST(SensorModel, initVariableLengthModel)
{   
    const int N_DIMS_z = 3;
    VariableLengthSensorModel model(N_DIMS_z);
    EXPECT_EQ(model.h(Vec_x::Zero()), Vec_x::Zero());

    Vec_x x{1,2};
    EXPECT_EQ(model.h(x), x);
}


} // variable_length_sensor_model_test
