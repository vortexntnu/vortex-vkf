#include <gtest/gtest.h>
#include <models/sensor_model.hpp>
#include <probability/multi_var_gauss.hpp>
#include "test_models.hpp"

namespace simple_sensor_model_test {

using Measurement = typename SimpleSensorModel::Measurement;
using State = typename SimpleSensorModel::State;
using Mat_xx = typename SimpleSensorModel::Mat_xx;


TEST(SensorModel, initSimpleModel)
{   
    SimpleSensorModel model;
    EXPECT_EQ(model.h(State::Zero()), State::Zero());

    State x{1,2};
    EXPECT_EQ(model.h(x), x);
}

TEST(SensorModel, predictSimpleModel)
{
    SimpleSensorModel model;
    vortex::prob::MultiVarGauss<2> x_est{State::Zero(), Mat_xx::Identity()};
    vortex::prob::MultiVarGauss<2> pred = model.pred_from_est(x_est);
    EXPECT_EQ(pred.mean(), State::Zero());
    EXPECT_TRUE(pred.cov().isApprox(Mat_xx::Identity()*1.1));

    pred = model.pred_from_state(x_est.mean());
    EXPECT_TRUE(pred.cov().isApprox(Mat_xx::Identity()*0.1));
}


} // simple_sensor_model_test