#include <gtest/gtest.h>
#include <iostream>

#include <filters/ekf.hpp>
#include <probability/multi_var_gauss.hpp>
#include "test_models.hpp"

using namespace vortex::filters;
using namespace vortex::models;
using namespace vortex::prob;


const int N_DIMS_x = SimpleDynamicModel::N_DIM_x;
const int N_DIMS_z = SimpleSensorModel::N_DIM_z;
using DynModI = DynamicModelI<N_DIMS_x>;
using SensModI = SensorModelI<N_DIMS_x, N_DIMS_z>;
using Vec_x = typename DynModI::Vec_x;
using Mat_xx = typename DynModI::Mat_xx;
using Vec_z = typename SensModI::Vec_z;
using Gauss_x = typename DynModI::Gauss_x;
using Gauss_z = typename SensModI::Gauss_z;

TEST(EKF, Simple) {

    SimpleDynamicModel dynamic_model;
    SimpleSensorModel sensor_model;
    EKF<SimpleDynamicModel, SimpleSensorModel> ekf(dynamic_model, sensor_model);

    // Initial state
    Gauss_x x({0, 0}, Mat_xx::Identity());

    // Predict
    auto pred = ekf.predict(x, 0.1);
    Gauss_x x_est_pred = pred.first;
    Gauss_z z_est_pred = pred.second;

    // Update
    Vec_z z = {1, 1};
    ekf.update(x_est_pred, z_est_pred, z);

    // Check that the state is close to zero
    // ASSERT_TRUE(x.isMuchSmallerThan(Vec_x::Ones()));
}
